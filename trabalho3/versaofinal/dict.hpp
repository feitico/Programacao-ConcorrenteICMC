#define MAX_SHORT_SIZE 6
#define MAX_COMPOUND_SIZE 46
#define NOT_FOUND 999999

/* Dicionario */
class Dict 
{
    private:
        char** words; //Palavras do dicionario
        int* marked; //Indica se foi marcado ou nao
        int* qtdMarkedLength; //Indica qntas palavras de determinado tamanho já foram marcadas
        int qtd; // Qtd de palavras no dicionario
        int qtdMarked; // Qtd de palavras marcadas
        int maxlength; // Tamanho maximo das palavras no dicionario
    public:
        Dict();
        ~Dict();

        void init(int qtd, int maxlength);

        void insert(int pos, char* word);
        int markWord(char* word);
        int markPos(int pos);

        int search(char* word);
        void print(int marked);
        
        char** getWords();

        int* getMarked();
        void setMarked(int* newMarked);

        int getQtd();
        int getQtdMarked(); //Retorna quantas palavras foram marcadas
        int getQtdMarkedLength(int length);
        int getMaxWordLength();
};


