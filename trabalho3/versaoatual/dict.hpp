#define MAX_SHORT_SIZE 6
#define MAX_COMPOUND_SIZE 46

/* Dicionario */
class Dict 
{
    private:
        char** words; //Palavras do dicionario
        int* marked; //Indica se foi marcado ou nao
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
        void print();
        
        char** getWords();
        int* getMarked();
        int getQtd();
        int getQtdMarked(); //Retorna quantas palavras foram marcadas
        int getMaxWordLength();
};


