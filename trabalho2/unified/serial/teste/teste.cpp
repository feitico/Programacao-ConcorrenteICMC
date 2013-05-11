#include <string>
#include <iostream>
#include <fstream>

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

int main(int argc, char* argv[]) {
	string line, frase, palavra;
	int found=string::npos, last;
    int max=0, num;
    ifstream in(argv[1]);
    int numprocs = 10;
    int proc, i;
    string str[numprocs];
    frase="";
	while(!in.eof()) {
        if(found == string::npos) {
            getline(in, line);
            found = line.find_first_of(" .!?");
            last = 0;
        }
        proc = 1;
        // Enviando mensagens
		while(1) {
            if(found != string::npos) {
                if(last != 0)
                    last++;

                palavra = trim(line.substr(last, found-last));
                if(!palavra.empty()) {
                    cout << "sent to " << proc++ << ": " << palavra << endl;
                    frase.append(palavra);
                    if(proc == numprocs)
                        break;
                }

                if(line[found] != ' ') {
                    cout <<  "sent to " <<proc++ << " frase : " << frase << endl;
                    frase = "";
                    if(proc == numprocs)
                        break;
                }
                last = found;
                found = line.find_first_of(" .!?", found+1);
            } else {
                if(last == 0) {
                    palavra = trim(line.substr(last, found-last));
                    cout <<  "sent to " <<proc++ << ": " << palavra << endl;
                    if(proc == numprocs)
                        break;
                } else {
                    palavra = trim(line.substr(last+1, found-last));
                    cout <<  "sent to " <<proc++ << ": " << palavra << endl;
                    if(!palavra.empty())
                        frase.append(palavra);
                    if(proc == numprocs)
                        break;
                }
                getline(in,line);
                found = line.find_first_not_of(" .!?");
                last = 0;
            }
        }

        for(i=1; i<proc; i++)
            cout << "receive from " << i << endl; 
	}
    in.close();
	return 0;
}
