# News_QA
#### 使用kaggle資料集，通過spark做處理之後，餵給QA model
---
### 資料觀察&前處理
1. 資料下載
將[kaggle news的資料集](https://www.kaggle.com/datasets/nikhiljohnk/news-popularity-in-multiple-social-media-platforms)從kaggle下載後
將train_file.csv檔放入colab資料夾

2. 資料觀察
```
news_data = sc.parallelize(news_arr)
news_data.count()
```
![image](https://user-images.githubusercontent.com/44884255/161244972-c24ace47-b236-4a69-a4d6-c674ab400519.png)

資料總數為 5萬筆

```
news_data.take(10)
```
![image](https://user-images.githubusercontent.com/44884255/161244665-8bc596f1-aa6a-4f18-a5b3-ef7b93af8aa6.png)
觀察一下資料的結構

3. 資料處理
```
news_trim = news_data.map(lambda x : [x[4],x[2]])
```
將需要的資料Headline拿出來Topic備用
```
news_trim.take(10)
```
觀察取出的資料是否正確
![image](https://user-images.githubusercontent.com/44884255/161245646-7d2cdc67-d321-4672-ab46-2de654585d1f.png)

4. 模型載入
由[Hugging Face](https://huggingface.co/)載入用來做QA的Roberta
```
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
```

使用裡面的context嘗試做預測
```
QA_input = {
    'question': 'Which company develop AR headset?',
    'context': "Microsoft’s AR headset is being made available to developers, along with three new games – including one based on Conker’s Bad Fur Day."
}
res = nlp(QA_input)
```
查看結果
```
res["answer"]
```
![image](https://user-images.githubusercontent.com/44884255/161246431-3300c63f-40fa-448e-bc58-b3ff54692c98.png)5

5. 分詞
因為想要輸入問句就可以輸出答案，所以需要將輸入的問句做處理
先載入nltk套件
```
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
```
接下來撰寫一個function分詞，並且拿出是名詞的字
```
def get_noun(sentence):
  token = nltk.word_tokenize(sentence)
  token = nltk.pos_tag(token)
  res = []
  for i in token:
    if i[0].lower() == "news":
      continue
    if "NN" in i[1]:
      res.append(i[0])  
  return res
```
查看結果
```
sentence = 'News of Ukraine'
noun_list = get_noun(sentence)
print(noun_list)
```
![image](https://user-images.githubusercontent.com/44884255/161250535-8360a72b-3861-48e9-964e-dd2e0d22ba83.png)

6.使用spark將相關的句子取出並且丟入QA model
```
context = news_trim.map(lambda x : have_word(x)).filter(lambda y: y != None ).collect()
answers = []
for c in context:
  QA_input={'question': sentence, 'context':c[1]}
  answers.append(nlp(QA_input)["answer"])
  print(nlp(QA_input)["answer"])
```
用來map的have_word function，主要是會將match問句名詞的句子回傳
```
def have_word(sentence):
  for i in noun_list:
    if i in sentence[1]:
      return sentence
  else:
    return None
```
最後將答案的排名輸出
```
sc.parallelize(answers).map(lambda x: (x,1)).reduceByKey(lambda x,y:x+y).sortBy(lambda x:x[1],ascending=False).collect()
```
![image](https://user-images.githubusercontent.com/44884255/161252185-467b7ceb-c3b4-43fb-9df0-a258604f32f3.png)

