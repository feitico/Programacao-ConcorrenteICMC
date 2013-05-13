#!/bin/bash
for((i = $1; i<= $2; i++)) do
	./split $i ../../shakespeare.txt
	TEMPO=$(./timediff.bash $3 './small $i')
	echo "$i,$TEMPO"

	#result=$(awk -vn1="$TEMPO" -vn2="$TEMPO_A" 'BEGIN{print (n1>n2)?1:0 }')
	#echo "$result"
	#if [ "$result" -eq 1 ]; then
	#	let k=$k-1
	#fi
	
	#if [ "$k" -eq "0" ]; then
	#	break;
	#fi
	#TEMPO_A=$TEMPO
done
rm -rf split*.txt # apaga os split*.txt 
