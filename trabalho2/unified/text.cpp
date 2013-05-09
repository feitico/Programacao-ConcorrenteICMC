#include <string>
#include <iostream>

using namespace std;


string trim(const string &str)
{
	size_t s = str.find_first_not_of(" \n\r\t");
	size_t e = str.find_last_not_of (" \n\r\t");

	if(( string::npos == s) || ( string::npos == e))
		return "";
	else
		return str.substr(s, e-s+1);
}

int main() {
	string str, frase, palavra;
	int found, last;
	while(1) {
		getline(cin,str);
		frase="";
		found = str.find_first_of(" .!?\n");
		last = 0;
		while(found != string::npos) {
			if(last != 0) {
				last++;
			} 

			palavra = trim(str.substr(last, found-last));
			if(!palavra.empty())
				frase.append(palavra);

			if(str[found] != ' ') {
				cout << "f: " << frase << endl;
				frase = "";
			} 

			last = found;
			found = str.find_first_of(" .!?\n", found+1);
		}
		if(last == 0) {
			palavra = trim(str.substr(last, found-last));
			if(!palavra.empty())
				frase.append(palavra);
		} else {
			palavra = trim(str.substr(last+1, found-last));
			if(!palavra.empty())
				frase.append(palavra);	
		}

		if(!frase.empty())
			cout << "f: " << frase << endl;
			
	}
	return 0;
}
