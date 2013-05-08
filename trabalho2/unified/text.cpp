#include <string>
#include <iostream>

using namespace std;

int main() {
	string str, phase;
	int found, last;
	while(1) {
		getline(cin,str);
		phase="";
		found = str.find_first_of(" .!?\n");
		last = 0;
		while(found != string::npos) {
			if(last != 0) {
				last++;
			} 

			phase.append(str.substr(last, found-last));

			if(str[found] != ' ') {
				cout << "f: " << phase << endl;
				phase = "";
			} 

			last = found;
			found = str.find_first_of(" .!?\n", found+1);
		}
		if(last == 0) {
			phase.append(str.substr(last, found-last));
		} else {
			if(!str.substr(last+1,found-last).empty()) {
				phase.append(str.substr(last+1,found-last));
			}
		}

		if(!phase.empty())
			cout << "f: " << phase << endl;
			
	}
	return 0;
}
