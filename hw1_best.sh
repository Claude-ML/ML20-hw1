if [ $# != 2 ]; then
    echo "Usage: bash hw1.sh <input file> <output file>"
        exit 1
fi

echo "Input file: " $1
echo "Output file: " $2
python3 test1.py $1 $2
