
import pandas as pd
print("Loading ...")
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
output_dir = "checkpoint-6000"
model = AutoModelForSeq2SeqLM.from_pretrained(output_dir )
tokenizer = AutoTokenizer.from_pretrained(output_dir)
def generate_sentences(data):
    # Sample input text
    decoded_outputs = []
    prefix = "translate German to English: "
    for i in range(data['de'].shape[0]):
        inputs = data['de'].iloc[i]

        # Tokenize the sample input text
        inputs = tokenizer(prefix + inputs, return_tensors="pt", max_length=512, truncation=True, padding=True)
        # Generate translations
        generated_output = model.generate(inputs['input_ids'],max_new_tokens = 512)

        # Decode the generated output
        decoded_output = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        decoded_outputs.append(decoded_output)
    print(decoded_output)
    return decoded_outputs
while True : 
    print("1. For translating the sentence : ")
    print("2. For translating the csv file (name csv file as test.csv)")
    try : 
        inp = int(input("Enter your input : "))
    except ValueError:
        print("Incorrect input")
        continue

    print("-----------------------------------------------------------")

    if (inp == 1):
        sentence = input("Enter your german sentence : ")
        inputs = tokenizer(sentence, return_tensors="pt", max_length=1024, truncation=True)
        # Generate predictions
        outputs = model.generate(**inputs,max_new_tokens=512)

        # Decode the output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True )
        print("Generated output:", output_text)
    
    else : 
        data = pd.read_csv('test.csv')
        test_gen = generate_sentences(data)
        output_csv = pd.DataFrame({'de' : data['de'],'en':test_gen})
        output_csv.to_csv('group_37_2c_output.csv')
    