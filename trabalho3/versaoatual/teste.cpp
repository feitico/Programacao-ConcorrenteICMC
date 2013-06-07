#include <iostream>
#include <cstdlib>
#include <sys/time.h>

using namespace std;

void seed_rand(int thread_n, struct drand48_data *buffer)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	srand48_r(tv.tv_sec * thread_n + tv.tv_usec, buffer);
}

double myrand(struct drand48_data &buffer) {
    double temp;
    drand48_r(&buffer, &temp);
    return temp;
}

int main() {
	struct drand48_data buffer;
	seed_rand(1, &buffer);

	int qtd[27];

	for(int i=0; i<27; i++)
		qtd[i] = 0;

	for(int i=0; i<10000000; i++)
		qtd[(int) (myrand(buffer) * 5 + 1)]++;

	for(int i=0; i<27; i++)
		cout << i << " - " << qtd[i] << endl;

	return 0;
}
