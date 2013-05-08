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
			if(last != 0)
				last++;

			if( (found - last >= 1)	|| (str[found] != ' ')) 
				phase.append(str.substr(last, found));

			if(str[found] != ' ') {
				cout << "phase1: " << phase << endl;

				phase = "";
			}

			last = found;
			found = str.find_first_of(" .!?\n", found+1);
		}
		if(phase
		if(last != 0) {
			phase.append(str.substr(last+1, found));
			cout << "phase2" << phase << endl;
		}
		if(phase.empty() && last == 0) { 
			cout << phase.empty() << " " << last << endl;
			cout << "phase3 " << str << endl;
		}
			
	}
	return 0;
}
