def get_change_word(txt,i):
    p = i
    while p>=0 and txt[p] != " ":
        p -= 1
    # print(p)
    return txt[p+1:i+1]
if __name__ == "__main__":
    # print(get_change_word("asda lp",6))
    # print(chr(36))
    # for i in range(33,127):
    #     print(i," ",chr(i))
    a = [chr(i) for i in range(33,127)]
    print(a)