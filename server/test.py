def get_change_word(txt,i):
    p = i
    while p>=0 and txt[p] != " ":
        p -= 1
    # print(p)
    return txt[p+1:i+1]
if __name__ == "__main__":
    # print(get_change_word("asda lp",6))
    s = "abcdefg"
    print(s+"test") 
    lzl