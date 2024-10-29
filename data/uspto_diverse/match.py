import json



def main():
    query = 'CC(C)(C)OC(=O)N1CCC(NCc2ccc(Cl)cc2)CC1'
    rfile = 'raw_test.json'
    with open(rfile) as rf:
        refs = json.load(rf)
    for key in refs.keys():
        cleaned_key = key
        cleaned_key = cleaned_key.replace(' ', '')
        if query == cleaned_key:
            print(key)
            return

if __name__ == "__main__":
    main()