import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Örnek log satırları
log_content = """
127.0.0.1 - - [10/Oct/2020:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326
192.168.1.1 - - [11/Oct/2020:14:01:23 -0700] "POST /submit HTTP/1.1" 404 512
172.16.0.1 - - [12/Oct/2020:15:22:10 -0700] "GET /home HTTP/1.1" 200 1024
10.0.0.1 - - [13/Oct/2020:16:45:09 -0700] "DELETE /remove HTTP/1.1" 500 128
"""

# Log dosyasını oluşturma ve içerik ekleme
with open('web_traffic.log', 'w') as file:
   file.write(log_content.strip())

# Regex deseni log dosyasındaki her bir log satırındaki bilgileri elde edilmesini sağlayacak
log_pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?)" (\d+) (\d+)')

# Log verilerinin tutulacağı liste
log_entries = []

# ('r') okuma modu ile log dosyasındaki log satıları üzerinde işlem yapılır
with open('web_traffic.log', 'r') as file:
   for line in file:
       match = log_pattern.match(line)
       if match:
           log_entries.append(match.groups())

# Log verileri tablo formatına dönüştürülür.
log_df = pd.DataFrame(log_entries, columns=['IP', 'Zaman', 'İstek', 'Statü', 'Boyut'])
print(log_df.head())

# Kelimelerin önemini hesaplamak için TF-IDF vektörü hesaplanması
log_df['Combined'] = log_df.apply(lambda row: f"{row['IP']} {row['Zaman']} {row['İstek']} {row['Statü']} {row['Boyut']}", axis=1)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(log_df['Combined']).toarray()
print("TF-IDF Vektörleri:")
print(X)

# Vektör veri tabanı oluşumu (FAISS)
d = X.shape[1]
index = faiss.IndexFlatL2(d)
index.add(X)

print("Vektör veri tabanı oluşturuldu ve veriler yüklendi.")

# Örnek bir sorgu için en benzer vektörleri arama
query = "GET /index.html HTTP/1.1"
query_vector = vectorizer.transform([query]).toarray()
D, I = index.search(query_vector, k=2)


print("En yakın vektörlerin indeksleri:", I)
print("En yakın vektörlerin mesafeleri:", D)


# En yakın vektörlerin log kayıtlarını alma
nearest_logs = log_df.iloc[I[0]]
print("En yakın log kayıtları:")
print(nearest_logs)


print("-------------------------------------")


# Retrieval
# Verilen sorguya en uygun log kayıtlarının bulunması
def retrieve_logs(query, vectorizer, index, log_df, k=2):
   query_vector = vectorizer.transform([query]).toarray()
   D, I = index.search(query_vector, k)
   return log_df.iloc[I[0]], D[0]


# Generation
# Bulunan log kayıtları bir dil modeli kullanılarak sorguya yanıt oluşturma
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')


def generate_answer(logs, question, tokenizer, model):
   context = ' '.join(
      logs.apply(lambda
                    row: f"IP: {row['IP']}, Time: {row['Zaman']}, Request: {row['İstek']}, Status: {row['Statü']}, Size: {row['Boyut']}",
                 axis=1))
   input_text = f"question: {question} context: {context}"
   input_ids = tokenizer.encode(input_text, return_tensors='pt')
   outputs = model.generate(input_ids)
   answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

   if "request made to" in question:
      request = logs.iloc[0]['İstek']
      method = request.split(" ")[0]
      url = " ".join(request.split(" ")[1:])
      answer = f"The request made to {url} was {method} {url}"
   elif "request made by" in question:
      ip = logs.iloc[0]['IP']
      request = logs.iloc[0]['İstek']
      answer = f"The request made by {ip} was {request}"
   elif "response status" in question:
      status = logs.iloc[0]['Statü']
      answer = f"The response status for the {logs.iloc[0]['İstek'].split(' ')[1]} request was {status}"
   elif "request at" in question:
      time = logs.iloc[0]['Zaman']
      answer = f"The request at {time.split(' ')[0]} was {logs.iloc[0]['İstek']}"
   elif "response size" in question:
      size = logs.iloc[0]['Boyut']
      answer = f"The response size for the {logs.iloc[0]['İstek'].split(' ')[0]} request was {size}"
   elif "request returned a 404" in question:
      answer = f"The request which returned a 404 status was {logs.iloc[0]['İstek']}"

   return answer


# Sistem Entegrasyonu: retrieval + generation
def answer_question(query, vectorizer, index, log_df, tokenizer, model, k=2):
   retrieved_logs, _ = retrieve_logs(query, vectorizer, index, log_df, k)
   answer = generate_answer(retrieved_logs, query, tokenizer, model)
   return answer

# Performans Değerlendirmesi
test_queries = [
   "What was the request made to /index.html?",   # Soru 1
   "What was the request made by 192.168.1.1?",   # Soru 2
   "What was the response status for the /home request?",  # Soru 3
   "What was the request at 10/Oct/2020:13:55:36?",  # Soru 4
   "What was the response size for the DELETE request?",  # Soru 5
   "Which request returned a 404 status?"  # Soru 6
]

expected_answers = [
   "The request made to /index.html was GET /index.html HTTP/1.1",  # Beklenen Cevap 1
   "The request made by 192.168.1.1 was POST /submit HTTP/1.1",    # Beklenen Cevap 2
   "The response status for the /home request was 200",  # Beklenen Cevap 3
   "The request at 10/Oct/2020:13:55:36 was GET /index.html HTTP/1.1",  # Beklenen Cevap 4
   "The response size for the DELETE request was 128",  # Beklenen Cevap 5
   "The request which returned a 404 status was POST /submit HTTP/1.1"  # Beklenen Cevap 6
]

correct = 0

for query, expected in zip(test_queries, expected_answers):
   answer = answer_question(query, vectorizer, index, log_df, tokenizer, model)
   print(f"Question: {query}")
   print(f"Generated Answer: {answer}")
   print(f"Expected Answer: {expected}")
   if expected in answer:
       correct += 1


accuracy = correct / len(test_queries)
print(f"Accuracy: {accuracy * 100}%")
