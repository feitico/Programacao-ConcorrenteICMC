#include<stdio.h>
#include<malloc.h>
 
typedef struct trie
{
        int words;
        int prefixes;
        struct trie *edges[26];
}trie;
 
trie * initialize(trie *node)
{
        if(node==NULL)
        {
                node=(trie *)malloc(sizeof(trie));
                node->words=0;
                node->prefixes=0;
                int i;
                for(i=0;i<26;i++)
                        node->edges[i]=NULL;
                return node;
        }
 
}
 
trie * addWord(trie *ver,char *str)
{
        printf("%c",str[0]);
        if(str[0]=='\0')
        {
                ver->words=ver->words+1;
        }
        else
        {
 
                ver->prefixes=(ver->prefixes)+1;
                char k;
                k=str[0];
                str++;
                int index=k-'a';
                if(ver->edges[index]==NULL)
                {
                        ver->edges[index]=initialize(ver->edges[index]);
                }
                ver->edges[index]=addWord(ver->edges[index],str);
        }
        return ver;
}
 
int countWords(trie *ver,char *str)
{
        if(str[0]=='\0')
                return ver->words;
        else
        {
                int k=str[0]-'a';
                str++;
                if(ver->edges[k]==NULL)
                        return 0;
                return countWords(ver->edges[k],str);
        }
}
 
int countPrefix(trie *ver,char *str)
{
        if(str[0]=='\0')
                return ver->prefixes;
        else
        {
                int k=str[0]-'a';
                str++;
                if(ver->edges[k]==NULL)
                        return 0;
                return countPrefix(ver->edges[k],str);
        }
}
 
 
 
int main()
{
        trie *start=NULL;
        start=initialize(start);
        int ch=1;
        while(ch)
        {
 
                printf("\n 1. Insert a word ");
                printf("\n 2. Count words");
                printf("\n 3. Count prefixes");
                printf("\n 0. Exit\n");
                printf("\nEnter your choice: ");
                scanf("%d",&ch);
                char input[1000];
                switch(ch)
                {
                        case 1:
                                printf("\nEnter a word to insert: ");
                                scanf("%s",input);
                                start=addWord(start,input);
                                break;
                        case 2:
                                printf("\nEnter a word to count words: ");
                                scanf("%s",input);
                                printf("\n%d",countWords(start,input));
                                break;
                        case 3:
                                printf("\nEnter a word to count prefixes: ");
                                scanf("%s",input);
                                printf("\n%d",countPrefix(start,input));
                                break;
                }
        }
        return 0;
}
