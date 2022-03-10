# This shell script splits a given file including the whole file list
#  into several files separately including sub file list
#     usage: splitFile.sh INPUTFILE [SPLITNUM]
#
# The default value of splitNum is 40
#
# Copyright (c) Su Rongfeng
# Last Updated 12/6/2012
#

SPLITNUM=40

if [ $# -ne 2 ]
then
    if [ $# -ne 1 ]
    then
    	echo "Usage: createSubFileList.sh INPUTFILE [SPLITNUM]"
    fi
fi

# Check Source Exists
SRC=$1
if [ ! -f $SRC ]
then
	echo "the source file $SRC does not exist"
fi

#set the SPLITNUM
if [ $# -eq 2 ]
then
	SPLITNUM=$2
fi

LINECOUNT=`awk 'END{print NR}' $SRC`      #the total line in the source file ".scp"

#caculate the number of file names in each splitting file
LINEPERFILE=`echo "scale=2; $LINECOUNT/$SPLITNUM" | bc`
LINEPERFILE=`echo $LINEPERFILE | awk '{print int($1)==$1?int($1):int(int($1+1))}'`

#the basename of the target file
#train.scp --> train_sub
TGR=`echo $SRC | sed -e 's/.scp/_sub/'`
COUNT=1
FROM=1
TO=$LINEPERFILE

if [ $LINECOUNT -ge $SPLITNUM ]
then
    while [ $COUNT -le $SPLITNUM ]
    do
#get the lines from $FROM to $TO and put them into $TGR$COUNT".scp"
        sed -n ''"$FROM"','"$TO"'p' $SRC > $TGR$COUNT".scp"
        FROM=`expr $FROM + $LINEPERFILE`
        TO=`expr $TO + $LINEPERFILE`
        if [ $TO -gt $LINECOUNT ]
        then
            TO=$LINECOUNT
        fi
        COUNT=`expr $COUNT + 1`
    done
else
    echo "The SPLITNUM is too large!"
    echo "Usage: createSubFileList.sh INPUTFILE [SPLITNUM]"
fi
