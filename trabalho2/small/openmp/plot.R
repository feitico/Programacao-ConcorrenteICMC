time = read.csv("experimento.txt", header=FALSE)
jpeg("small_time_thread.jpg")
plot(time, main="Tempo de Execução vs Número de Threads", xlab="Numero de Threads", ylab="Tempo de Execução", log="xy")
dev.off();
