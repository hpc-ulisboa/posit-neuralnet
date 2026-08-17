#ifndef ADAM_HPP
#define ADAM_HPP

#include <sstream>
#include <stdexcept>

#include "../layer/Parameter.hpp"
#include "../optimizer/Optimizer.hpp"
#include "../tensor/StdTensor.hpp"
#include "../tensor/matrix.hpp"      // fused()

using namespace sw::unum;

// AdamOptions holds the configured hyperparameters as double, i.e. as the caller
// wrote them, so that what reaches the optimizer is the requested value and not
// one already mangled by a conversion the caller never asked for.
//
// Everything after that runs in T. beta^t, the bias corrections and the moment
// recurrences are all evaluated in the parameter format, exactly as they would be
// on posit hardware. Carrying them in double would make the optimizer behave
// better than the format it is supposed to be measuring, which is the opposite of
// what this library is for.
//
// That leaves the failure the conversion itself can cause: posit<8,2> rounds
// 0.999 to exactly 1, so 1 - beta_2 is 0, the bias correction divides by zero and
// the model is NaR after one step. This is a real property of the format, not
// something to paper over, so the constructor checks the converted values and
// refuses to build an optimizer that cannot work.
//
// The struct stays writable through options(), as in SGD, so a training loop can
// retune the optimizer between steps. The converted values are cached, so step()
// reconverts and rechecks whatever changed before using them.
template <typename T>
struct AdamOptions {
    AdamOptions(double _learning_rate, double _beta_1=0.9, double _beta_2=0.999, double _eps=1e-8) :
        learning_rate(_learning_rate),
        beta_1(_beta_1),
        beta_2(_beta_2),
        eps(_eps),
        t(0)
    { }

    double learning_rate;
    double beta_1;
    double beta_2;
    double eps;
    size_t t;
};

template <typename T>
class Adam : public Optimizer<T> {
public:
    Adam(std::vector<Parameter<T>> parameters0, AdamOptions<T> options0) :
        Optimizer<T>(parameters0),
        _options(options0),
        _synced_t(options0.t),
        _beta_1(T(options0.beta_1)),
        _beta_2(T(options0.beta_2)),
        // 1 - beta in T, not T(1.0 - beta): the subtraction is arithmetic and so
        // belongs in the parameter format like every other operation here.
        _one_minus_beta1(T(1) - T(options0.beta_1)),
        _one_minus_beta2(T(1) - T(options0.beta_2)),
        _learning_rate(T(options0.learning_rate)),
        _eps(T(options0.eps)),
        _beta1_t(T(1)),
        _beta2_t(T(1)),
        _bias_corr1(T(1)),
        _bias_corr2(T(1))
    {
        check_beta(options0.beta_1, _beta_1, _one_minus_beta1, "beta_1");
        check_beta(options0.beta_2, _beta_2, _one_minus_beta2, "beta_2");
        check_learning_rate(options0.learning_rate, _learning_rate);

        for (auto const& p : this->_parameters) {
            _m.push_back(StdTensor<T>(p.weight.shape()));
            _v.push_back(StdTensor<T>(p.weight.shape()));
            _m.back().clear();
            _v.back().clear();
        }
    }

    void step() {
        // Pick up anything the caller wrote into options() since the last step.
        sync_options();

        // The bias correction only depends on t, so it is computed once per step
        // instead of once per parameter. beta^t is carried forward incrementally,
        // which is one multiplication per step instead of a pow() in O(t).
        //
        // All of it in T: beta^t decays in the parameter format, and the bias
        // correction is the posit division the hardware would do.
        //
        // t and beta^t advance together, so the invariant sync_options() relies on
        // -- beta^t is beta multiplied into itself t times -- still holds after.
        _options.t++;
        _synced_t = _options.t;
        _beta1_t *= _beta_1;
        _beta2_t *= _beta_2;
        T const one(1);
        _bias_corr1 = one / (one - _beta1_t);
        _bias_corr2 = one / (one - _beta2_t);

        Optimizer<T>::step();
    }

    AdamOptions<T>& options() { return _options; }

    std::vector<StdTensor<T>> const& moments_m() const { return _m; }
    std::vector<StdTensor<T>> const& moments_v() const { return _v; }

private:
    // options() hands out a mutable reference, as SGD's does, and the examples use
    // it for a learning rate schedule. Since the conversions are cached, a write
    // through that reference would otherwise change nothing at all, silently. So
    // every step restores the invariant that the cached values are what the
    // options currently convert to, which is what the rest of the class assumes.
    //
    // Stated that way rather than as "did the caller assign something", the check
    // compares the converted values, not the doubles: two doubles that round to
    // the same posit leave the optimizer's behaviour identical, so there is
    // nothing to redo. It costs four conversions per step, against a whole
    // parameter update's worth of posit arithmetic.
    //
    // The constructor's checks are repeated here rather than skipped: a learning
    // rate or beta assigned later can be just as unrepresentable as one passed in,
    // and the same exception is a better answer than a training run that quietly
    // stops moving.
    void sync_options() {
        T const lr(_options.learning_rate);
        if (lr != _learning_rate) {
            _learning_rate = lr;
            check_learning_rate(_options.learning_rate, _learning_rate);
        }

        T const eps(_options.eps);
        if (eps != _eps)
            _eps = eps;

        T const beta_1(_options.beta_1);
        T const beta_2(_options.beta_2);
        bool const beta1_changed = (beta_1 != _beta_1);
        bool const beta2_changed = (beta_2 != _beta_2);

        if (beta1_changed) {
            _beta_1 = beta_1;
            _one_minus_beta1 = T(1) - _beta_1;
            check_beta(_options.beta_1, _beta_1, _one_minus_beta1, "beta_1");
        }
        if (beta2_changed) {
            _beta_2 = beta_2;
            _one_minus_beta2 = T(1) - _beta_2;
            check_beta(_options.beta_2, _beta_2, _one_minus_beta2, "beta_2");
        }

        // t and the betas are what beta^t was accumulated from, so either one
        // moving leaves it stale. Rebuilding costs t multiplications, but only on
        // the step where something actually changed, and a beta is not something a
        // training loop retunes every step.
        if (beta1_changed || beta2_changed || _options.t != _synced_t)
            replay_beta_powers(_options.t);

        _synced_t = _options.t;
    }

    // beta^t from scratch, in T and with the same multiplication step() uses, so
    // the result is the value an uninterrupted run would have reached rather than
    // a cleaner one it never would have.
    void replay_beta_powers(size_t const t) {
        _beta1_t = T(1);
        _beta2_t = T(1);
        for (size_t i = 0; i < t; ++i) {
            _beta1_t *= _beta_1;
            _beta2_t *= _beta_2;
        }
    }

    void update_parameter(Parameter<T>& p, size_t const i) override {
        StdTensor<T>& w = p.weight;
        StdTensor<T>& m = _m[i];
        StdTensor<T>& v = _v[i];

        StdTensor<T> const& g = p.gradient;

        // The moment updates go through fused() (matrix.hpp), so each product is
        // accumulated exactly in the quire and rounded once, instead of the three
        // to four roundings of a naive element-wise evaluation. This matches SGD,
        // which already uses fused() throughout.
        fused(m, g, _beta_1, _one_minus_beta1);     // m = m*b1 + g*(1-b1)
        StdTensor<T> g2 = g * g;
        fused(v, g2, _beta_2, _one_minus_beta2);    // v = v*b2 + g2*(1-b2)

        // The weight update needs sqrt and division, so it stays element-wise
        using sw::unum::sqrt;
        for (size_t j = 0, n = w.size(); j < n; ++j) {
            T m_hat = m[j] * _bias_corr1;
            T v_hat = v[j] * _bias_corr2;
            w[j] = w[j] - (_learning_rate * m_hat) / (sqrt(v_hat) + _eps);
        }

        p.update();
    }

    static std::string format_name() {
        std::ostringstream s;
        s << "posit<" << T::nbits << "," << T::es << ">";
        return s.str();
    }

    // A beta is fatal when the conversion cannot keep it apart from 1: then
    // 1 - beta is 0, the moment never takes up the gradient, and beta^t stays 1
    // so the bias correction divides by zero. A beta that merely rounds is fine,
    // and beta = 0 is a legitimate (if unusual) choice, so neither is rejected.
    static void check_beta(double wanted, T const& beta, T const& one_minus, char const* name) {
        if (!beta.isnar() && !one_minus.iszero() && !one_minus.isnar())
            return;

        std::ostringstream msg;
        msg << "Adam: " << name << " = " << wanted << " becomes " << double(beta)
            << " in " << format_name() << ", leaving 1 - " << name << " = "
            << double(one_minus) << ". The moments would never take up the gradient "
               "and the bias correction would divide by zero. Use a wider optimizer "
               "format, or a " << name << " this one can keep apart from 1.";
        throw std::invalid_argument(msg.str());
    }

    // Only zero is fatal for the learning rate: no update would survive it. A
    // learning rate of 1 is large but perfectly well defined.
    static void check_learning_rate(double wanted, T const& lr) {
        if (!lr.isnar() && !lr.iszero())
            return;

        std::ostringstream msg;
        msg << "Adam: learning_rate = " << wanted << " becomes " << double(lr)
            << " in " << format_name() << ", so no weight would ever change. Use a "
               "wider optimizer format, or a learning rate this one can represent.";
        throw std::invalid_argument(msg.str());
    }

    AdamOptions<T> _options;
    size_t _synced_t;                       // the t that beta^t below was accumulated to
    std::vector<StdTensor<T>> _m;
    std::vector<StdTensor<T>> _v;
    T _beta_1, _beta_2;                     // converted once, used by fused()
    T _one_minus_beta1, _one_minus_beta2;
    T _learning_rate, _eps;
    T _beta1_t, _beta2_t;                   // beta^t, in the parameter format
    T _bias_corr1, _bias_corr2;             // written by step(), read by update_parameter()
};

#endif /* ADAM_HPP */
