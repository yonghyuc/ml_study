import json

if __name__ == "__main__":
    with open('../data/mpii_annotation.json') as json_file:
        data = json.load(json_file)
        for anno in data["annolist"]:
            print (anno)



