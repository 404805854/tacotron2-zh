from dataclasses import replace
from tacotron2.text import symbols


def main():
    str = input("id : ")
    out = ""
    arr = str.replace("[", "").replace("]", "").replace(" ", "").split(",")
    for iter in arr:
        try:
            index = int(iter)
            out += symbols[index]
        except:
            print("out of bound : ", iter)
            exit(-1)
    print(out)


if __name__ == "__main__":
    main()
