import spacy

# Load the trained model
model_path = r"C:\Users\Gouthum\Downloads\Project\batch_1\batch_1\MLDL\spacy_ner_model"
nlp = spacy.load(model_path)

# Sample test invoice text
test_text = """ 
Invoice no: 69721323
Date of issue: 05/07/2019

Seller: Murray-Eaton
773 Joseph Plains
West Nicoleville, AZ 46136

Tax Id: 936-71-8228

Client:
Cuevas, Reid and Hurst
98071 Daniel Heights
Careyside, MS 59400

Tax Id: 949-88-4885

IBAN: GB22XZGA27411153163644

ITEMS
No. Description Qty UM Net price Net worth VAT [%] Gross worth
1. Tie Dyeing Fluffy Rugs Anti-Skid 4,00 each 25,48 101,92 10% 112,11
2. Galaxy Butterfly Area Rugs 2,00 each 13,69 27,38 10% 30,12
3. Colorful Marble Printed Rugs 2,00 each 12,49 24,98 10% 27,48
4. Red Traditional Oriental Carpet 4,00 each 39,98 159,92 10% 175,91
5. YILONG Silk Carpets 4,00 each 2000,00 8000,00 10% 8800,00

SUMMARY
VAT [%] Net worth VAT Gross worth
10% 8 314,20 831,42 9 145,62
Total $ 8 314,20 $ 831,42 $9 145,62
"""

# Run model
doc = nlp(test_text)

# Print extracted entities
print("🔍 Entities detected:\n")
for ent in doc.ents:
    print(f"➤ {ent.label_:15} → {ent.text}")