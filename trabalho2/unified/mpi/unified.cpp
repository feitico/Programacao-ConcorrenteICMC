#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cctype>

#include "mpi.h"

#define MAX_NUM 8000
#define RAIZ_MAX_NUM 90

#define SMALL 0
#define LARGE 1

#define TAG_LENGTH 10
#define TAG_STR 11
#define TAG_ANSWER 12

#define NP 0 //Nao palindromo
#define PP 1 //Palindromo primo
#define PNP 2 //Palindromo nao primo
#define PFP 3 // Palavra ou frase palindroma

using namespace std;

int primo[MAX_NUM];
int atual_proc=0; //id do processador atual
int numprocs; // Quantidade de processadores

// Funcoes auxiliares
void init_crivo(int n); // Inicializa o crivo de erastotenes 
int isPrimo(int num); // Retorna 1 se for primo
int palindromo(string str); // Retorna 1 se for palindromo 
string trim(const string &str); // Remove os espacos em branco a direita e a esquerda da string
int proxProc(); // Retorna o id do proximo processo

int main(int argc, char* argv[]) {
	int i; //Contador
	int type; // Tipo de processamento: SMALL ou LARGE
	int length; // Otimiza o loop pára contar os ASCII de uma palavra

	int id; // id do processador
	MPI::Status status; // status do MPI_Recv

	if(argc != 3) {
		printf("Usage: ./unified type entrada.txt\n");
		exit(-1);
	}
	
	/* Indica se é para tratar o arquivo como large - palavra or small - palavra e por frase */
	type = atoi(argv[1]);

	MPI::Init(argc, argv); // Inicializa o MPI

	id = MPI::COMM_WORLD.Get_rank();

	numprocs = MPI::COMM_WORLD.Get_size();

    // Calculamos se eh primo somente no arquivo grande
    if(type == LARGE)
        init_crivo(MAX_NUM);

	// Verifica o id
	if(id == 0) {
		// Master node	
		ifstream entrada(argv[2], ifstream::in);
		int i,j, answer, proc;
        int word_count, last, found;
		string str[numprocs], line, palavra, frase;
        bool isfrase[numprocs]; // Indica se a mensagem mandada para o processador i eh uma frase
        vector<string> pp; //Vetor de palindromos primos
        vector<string> pnp; //Vetor de palindromos nao primos
        vector<string> pf; //Vetor de frases palindromas
        double start, end; // Tempo inicial e tempo final
        char nodeName[256]; //Nome do nó 
        int resultlen;
        string filename;
        stringstream ss;
                                                            
        MPI::Get_processor_name(nodeName, resultlen);
                                                            
        ss << "stats_" << nodeName <<  "-" << id << ".txt";
        ss >> filename;
        ofstream stats(filename.c_str());

        word_count = 0;
        found = string::npos;
        frase = "";
        start = MPI::Wtime(); // Tempo inicial
		while(!entrada.eof()) {
            if(type == SMALL) { // Processamento para arquivos pequenos
                if(found == string::npos) {
                    getline(entrada,line);
                    if(entrada.eof())
                        break;
                    found = line.find_first_of(" .!?");
                    last = 0;
                }
                proc = 1;
                //Enviando mensagens
                while(1) {
                    if(found != string::npos) {
                        if(last != 0)
                            last++;
                                                                                          
                        palavra = trim(line.substr(last, found-last));
                        if(!palavra.empty()) {
                		    word_count++;
                            // Envia o tamanho da palavra e a palavra
                    		length = palavra.size();
                    		// buffer, size, type, dest, tag
                    		MPI::COMM_WORLD.Send(&length, 1, MPI::INT, proc, TAG_LENGTH);
                    		MPI::COMM_WORLD.Send(palavra.c_str(), length, MPI::CHAR, proc, TAG_STR);
                         
                            isfrase[proc] = false;
                            str[proc] = palavra;
                            //cout << "d1 " << proc << " " << palavra << endl;
                            frase.append(palavra);
                            proc++;
                            if(proc == numprocs) {
                                last = found;
                                found = line.find_first_of(" .!?", found+1);
                                break;
                            }
                        }
                                                                                          
                        if(line[found] != ' ') {
                            if(word_count >= 2) {
                                // Envia o tamanho da frase e a frase
                                length = frase.size();
                                // buffer, size, type, dest, tag
                                MPI::COMM_WORLD.Send(&length, 1, MPI::INT, proc, TAG_LENGTH);
                                MPI::COMM_WORLD.Send(frase.c_str(), length, MPI::CHAR, proc, TAG_STR);
                        
                                isfrase[proc] = true;
                                str[proc] = frase; // str[proc] armaneza a frase enviada
                                //cout << "d2 " << proc << " " << frase << endl;
                                proc++;
                                if(proc == numprocs) {
                                    frase = "";
                                    word_count = 0;
                                    last = found;
                                    found = line.find_first_of(" .!?", found+1);
                                    break;
                                }
                            }
                            frase = "";
                            word_count=0;
                        }
                        last = found;
                        found = line.find_first_of(" .!?", found+1);
                    } else {

                        if(last == 0) {
                            palavra = trim(line.substr(last));
                            if(!palavra.empty()) {
                                word_count++;
                                // Envia o tamanho da palavra e a palavra
                                length = palavra.size();
                                // buffer, size, type, dest, tag
                                MPI::COMM_WORLD.Send(&length, 1, MPI::INT, proc, TAG_LENGTH);
                                MPI::COMM_WORLD.Send(palavra.c_str(), length, MPI::CHAR, proc, TAG_STR);   
                                

                                isfrase[proc] = false;
                                str[proc] = palavra;
                                //cout << "d3 " << proc << " " << palavra << endl;
                                frase.append(str[proc]);
                                proc++;
                                if(proc == numprocs)
                                    break;
                            }
                        } else {
                            palavra = trim(line.substr(last+1));
                            if(!palavra.empty()) {
                                word_count++;
                                // Envia o tamanho da palavra e a palavra
                                length = palavra.size();
                                // buffer, size, type, dest, tag
                                MPI::COMM_WORLD.Send(&length, 1, MPI::INT, proc, TAG_LENGTH);
                                MPI::COMM_WORLD.Send(palavra.c_str(), length, MPI::CHAR, proc, TAG_STR);    
                                
                                isfrase[proc] = false;
                                str[proc] = palavra;
                                //cout << "d4 " << proc << " " << palavra << endl;
                                frase.append(str[proc]);
                                proc++;
                                if(proc == numprocs)
                                    break;
                            }
                        }
                        getline(entrada,line);
                        if(entrada.eof())
                            break;
                        found = line.find_first_of(" .!?");
                        last = 0;
                    }
                }
                /*
                if(proc == numprocs) {
                    found = line.find_first_of(" .!?");
                    last = 0;
                }
*/
                
                // Envia a ultima possivel frase
                if( (proc != numprocs) && (!frase.empty()) && (word_count >= 2)) {
                    // Envia o tamanho da frase e a frase
                    length = frase.size();
                    // buffer, size, type, dest, tag
                    MPI::COMM_WORLD.Send(&length, 1, MPI::INT, proc, TAG_LENGTH);
                    MPI::COMM_WORLD.Send(frase.c_str(), length, MPI::CHAR, proc, TAG_STR);
                    
                    isfrase[proc] = true;
                    str[proc] = frase; // str[proc] armaneza a frase enviada
                    //cout << "d5 " << proc << " " << frase << endl;
                    frase = "";
                    word_count = 0;
                    proc++;
                }

                //Recebe a resposta de i processadores
                for(i=1; i<proc; i++) {
                    MPI::COMM_WORLD.Recv(&answer, 1, MPI::INT, i, TAG_ANSWER);
                    if(answer == PFP) {
                        if(isfrase[i] == true)
                            pf.push_back(str[i]);
                        else
                            pp.push_back(str[i]);       
                    }
                }
            } else { //Processamento para arquivos grandes
                //Le numprocs-1 palavras do arquivo e envia para os processadores
                for(i=1; i<numprocs; i++) { // Enviando mensagens
                    entrada >> str[i];
                    if(entrada.eof())
                        break;

                    // Envia o tamanho da string, seu indice no vetor e a string
                    length = str[i].size();
                    // buffer, size, type, dest, tag
                    MPI::COMM_WORLD.Send(&length, 1, MPI::INT, i, TAG_LENGTH); // Tamanho da string
                    MPI::COMM_WORLD.Send(str[i].c_str(), length, MPI::CHAR, i, TAG_STR); // string
                }

                //Recebe a resposta de i processadores
                for(j=1; j<i; j++) { // Recebendo mensagens
                    MPI::COMM_WORLD.Recv(&answer, 1, MPI::INT, j, TAG_ANSWER);
                    switch(answer) {
                        case PP: //Palavra Palindroma e Prima
                            pp.push_back(str[j]);
                            break;
                        case PNP: //Palavra Palindroma nao prima
                            pnp.push_back(str[j]);
                            break;
                    }
                }
            }
        }

        //Informa os processadores que nao existe mais palavras ou frases a serem processadas
        length = -1;
        for(i=1; i<numprocs; i++) {
            MPI::COMM_WORLD.Send(&length, 1, MPI::INT, i, TAG_LENGTH);
        }
        
        // Imprime os palindromos primos e os não primos                                    
        vector<string>::iterator it;
        if(type==LARGE)
        	cout << "Palindromos Primos (" << pp.size() << ")" << endl;
        else
        	cout << "Palindromos (" << pp.size() << ")" << endl;
                                                                                     
        for(it = pp.begin(); it != pp.end(); ++it)
        	cout << *it << endl;
        
        if(type==LARGE) {
        	cout << endl << "Palindromos Nao Primos (" << pnp.size() << ")" << endl;
        	for(it = pnp.begin(); it != pnp.end(); ++it)
        		cout << *it << endl;
        } else {
        	cout << endl << "Palindromos Frases (" << pf.size() << ")" << endl;
        	for(it = pf.begin(); it != pf.end(); ++it)
        		cout << *it << endl;
        }

        entrada.close();
        end = MPI::Wtime(); // Tempo inicial

        //Imprime estatistica do no master
        stats << nodeName << endl;
        stats << "time" << endl;
        stats << end - start << endl;
        stats.close();
	} else { // Worker nodes
		int length = 0;
		char trick[5000];
        int num;
        int answer;
        stringstream ss; // Usadao para concatenar o id e com outras strings
        string filename; // Nome do arquivo

        int stat_pp=0; //numero de palavras palindromas e primas
        int stat_pnp=0; //numero de palavras palindromas e nao primas
        int stat_pfp=0; //numero de palavras e frase palindromas
        int stat_np=0; //numero de palavras ou frase nao palindromas
        char nodeName[256];
        int resultlen;
        double start, end;

        MPI::Get_processor_name(nodeName, resultlen);

        ss << "stats_" << nodeName <<  "-" << id << ".txt";
        ss >> filename;
//        cout << id << "filename: " << filename << endl;

        ofstream stats(filename.c_str());

        start = MPI::Wtime(); //Inicio do trabalho do no worker
		while(1) {
			MPI::COMM_WORLD.Recv(&length, 1, MPI::INT, 0, TAG_LENGTH, status); // Recebe o tamanho da palavra ou frase
			if(length == -1) // Se for negativo, significa que acabou as palavras e frases do arquivo
				break;

			MPI::COMM_WORLD.Recv(trick, length, MPI::BYTE, 0, TAG_STR, status); // Recebe a string
            trick[length] = '\0'; // Termina a string
            string str(trick);
            //cout << "W" << id << " " << str.size() << " " <<  str << endl;
//            cout << id << str << endl;
            if(palindromo(str)) {
                //cout << "T" << type << endl;
            	if(type == LARGE) {
                    num=0;
                    length = str.size();
                    for(i=0; i<length; i++)
                        num += (int) str[i];
                                                                      
                    if(isPrimo(num) != 0) {
                        answer = PP;
                        stat_pp++;
                    } else {
                        answer = PNP;
                        stat_pnp++;
                    }
                } else {
                    answer = PFP;
                    stat_pfp++;
                }
            } else {
                answer = NP;
                stat_np++;
            }
//            cout << id << " " << answer <<  " " << length << " " << str << endl;
            MPI::COMM_WORLD.Send(&answer, 1, MPI::INT, 0, TAG_ANSWER);
		}
        end = MPI::Wtime(); // fim do trabalho do no worker

        stats << nodeName << endl;
        stats << "pp\tpnp\tpfp\tnp" << endl;
        stats << stat_pp << "\t" << stat_pnp << "\t" << stat_pfp << "\t" << stat_np << endl;
        stats << "time" << endl;
        stats << end - start << endl;
        stats.close();
	}

	MPI::Finalize(); // Finaliza o MPI
	return 0;
}

// Inicializa o crivo de erastotenes 
void init_crivo(int n) {
	int i, j;
	for(i=2; i<n; i++)
		primo[i] = i;
	
	int raiz = sqrt(n)+1;
	for(i=2; i<=raiz; i++) {
		if(primo[i] == i) {
			for(j=i+i; j<MAX_NUM; j+=i)
				primo[j] = 0;
		}
	}
}

// Retorna 1 se for primo
int isPrimo(int num) {
	return primo[num];
}

// Retorna 1 se for palindromo 
int palindromo(string str) {
	int i;
	int half;
	int length = str.size();

	if(length == 1)
		return 0;

	if(length % 2 == 0)
		half = length / 2;
	else
		half = (length / 2) + 1;


	for(i=0; i<=half; i++) {
		if(tolower(str[i]) != tolower(str[length-i-1])) {
			return 0;
		}
	}

	return 1;
}

// Remove os espacos em branco a direita e a esquerda da string
string trim(const string &str)
{
	size_t s = str.find_first_not_of(" \n\r\t");
	size_t e = str.find_last_not_of (" \n\r\t");

	if(( string::npos == s) || ( string::npos == e))
		return "";
	else
		return str.substr(s, e-s+1);
}

// Retorna o id do proximo processo
int proxProc() {
	atual_proc = (atual_proc+1) % numprocs;
	if(atual_proc == 0)
		atual_proc++;
	return atual_proc;
}
