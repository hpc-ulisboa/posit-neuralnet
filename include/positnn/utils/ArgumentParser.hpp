#ifndef ARGUMENTPARSER_HPP
#define ARGUMENTPARSER_HPP

struct ArgumentParser {
	ArgumentParser(int argc=0, char* argv[]=nullptr ) {
		if(argc < 2)
			return;

		save_path = argv[1];

		if(save_path.back() != '/')
			save_path += "/";
	}

	std::string join_paths(std::string head, const std::string& tail) {
        if (head.back() != '/') {
            head.push_back('/');
        }
        head += tail;
        return head;
    }

	std::string save_path;
};

#endif /* ARGUMENTPARSER_HPP */
