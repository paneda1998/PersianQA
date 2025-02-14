<span align="center">
    <a href="https://www.kaggle.com/sajjadayobi360/persianqa"><img alt="Kaggle" src="https://img.shields.io/static/v1?label=Kaggle&message=PersianQA&logo=Kaggle&color=20BEFF"/></a>
    <a href="https://huggingface.co/SajjadAyoubi/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=SajjadAyoubi&color=yellow"></a>
    <a href="https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/Demo.ipynb"><img src="https://img.shields.io/static/v1?label=Colab&message=Demo&logo=Google%20Colab&color=f9ab00"></a>
</span>

# PersianQA: a dataset for Persian Question Answering

Persian Question Answering (PersianQA) Dataset is a reading comprehension
dataset on [Persian Wikipedia](https://fa.wikipedia.org/). The crowd-sourced
dataset consists of more than 9,000 entries. Each entry can be either an
_impossible-to-answer_ or a question with one or more answers spanning in the
passage (the _context_) from which the questioner proposed the question.
Much like the SQuAD2.0 dataset, the impossible or _unanswerable_ questions can be
utilized to create a system which "knows that it doesn't know the answer".

Moreover, the dataset has 900 test data available. On top of that, the very
first models trained on the dataset, Transformers, are available online.

All the crowdworkers of the dataset are native Persian speakers. Also, it worth
mentioning that the contexts are collected from all categories of the Wiki
(Historical, Religious, Geography, Science, etc).

At the moment, each context has 7 pairs of questions with one answer and 3
impossible questions.

As mentioned before, the dataset is inspired by the famous SQuAD2.0 dataset and is
compatible with and can be merged into it. But that's not all, the dataset here
has some relative advantages to the original inspiration source, some of which
are listed below:

- Lengthier contexts
- Increased number of articles (despite having less data)
- More questions per contexts (7 comparing to 5)
- Including _informal ("Mohaaverei")_ entries
- More varied answers (names, locations, dates and more)

We train a baseline model which achieves an F1 score of 78 and an exact match
ratio of 52 on [ParsiNLU](https://github.com/persiannlp/parsinlu).

You can check out an online [Demo on Google Colab
](https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/Demo.ipynb).

## Dataset

### Access and Download

You can find the dataset under the [`dataset` directory](https://github.com/sajjjadayobi/PersianQA/tree/main/dataset) and use it like below:

```python
import read_qa # is avalible at src/read_ds.py
train_ds = read_qa('pqa_train.json')
test_ds  = read_qa('pqa_test.json')
```

Alternatively, you can also access the data through the HuggingFace🤗 datasets library.
For that, you need to install datasets using this command in your terminal:

```sh
pip install -q datasets
```

Afterwards, import `persian_qa` dataset using `load_dataset`:

```python
from datasets import load_dataset
dataset = load_dataset("SajjadAyoubi/persian_qa")
```

- The dataset is also available at [Kaggle](https://www.kaggle.com/sajjadayobi360/persianqa)

### Examples

|     Title     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Context                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |                    Question                     |               Answer                |
| :-----------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------: | :---------------------------------: |
| خوب، بد، زشت  |          خوب، بد، زشت یک فیلم درژانر وسترن اسپاگتی حماسی است که توسط سرجو لئونه در سال ۱۹۶۶ در ایتالیا ساخته شد. زبانی که بازیگران این فیلم به آن تکلم می‌کنند مخلوطی از ایتالیایی و انگلیسی است. این فیلم سومین (و آخرین) فیلم از سه‌گانهٔ دلار (Dollars Trilogy) سرجو لئونه است. این فیلم در حال حاضر در فهرست ۲۵۰ فیلم برتر تاریخ سینما در وب‌گاه IMDB با امتیاز ۸٫۸ از ۱۰، رتبهٔ هشتم را به خود اختصاص داده‌است و به عنوان بهترین فیلم وسترن تاریخ سینمای جهان شناخته می‌شود. «خوب» (کلینت ایستوود، در فیلم، با نام «بلوندی») و «زشت» (ایلای والاک، در فیلم، با نام «توکو») با هم کار می‌کنند و با شگرد خاصی، به گول زدن کلانترهای مناطق مختلف و پول درآوردن از این راه می‌پردازند. «بد» (لی وان کلیف) آدمکشی حرفه‌ای است که به‌خاطر پول حاضر به انجام هر کاری است. «بد»، که در فیلم او را «اِنجل آیز (اِینجل آیز)» (به انگلیسی: Angel Eyes) صدا می‌کنند. به‌دنبال گنجی است که در طی جنگ‌های داخلی آمریکا، به دست سربازی به نام «جکسون»، که بعدها به «کارسون» نامش را تغییر داده، مخفی شده‌است.           | در فیلم خوب بد زشت شخصیت ها کجایی صحبت می کنند؟ |    مخلوطی از ایتالیایی و انگلیسی    |
| قرارداد کرسنت | قرارداد کرسنت قراردادی برای فروش روزانه معادل ۵۰۰ میلیون فوت مکعب، گاز ترش میدان سلمان است، که در سال ۱۳۸۱ و در زمان وزارت بیژن نامدار زنگنه در دولت هفتم مابین شرکت کرسنت پترولیوم و شرکت ملی نفت ایران منعقد گردید. مذاکرات اولیه این قرارداد از سال ۱۹۹۷ آغاز شد و در نهایت، سال ۲۰۰۱ (۱۳۸۱) به امضای این تفاهم نامه مشترک انجامید. بر اساس مفاد این قرارداد، مقرر شده بود که از سال ۲۰۰۵ با احداث خط لوله در خلیج فارس، گاز فرآورده نشده میدان سلمان (مخزن مشترک با ابوظبی)، به میزان روزانه ۵۰۰ میلیون فوت مکعب (به قول برخی منابع ۶۰۰ میلیون فوت مکعب) به امارات صادر شود. این قرارداد مطابق قوانین داخلی ایران بسته شده‌ و تنها قرارداد نفتی ایران است که از طرف مقابل خود، تضمین گرفته‌است. اجرای این پروژه در سال ۱۳۸۴ با دلایل ارائه شده از سوی دیوان محاسبات ایران از جمله تغییر نیافتن بهای گاز صادراتی و ثابت ماندن آن در هفت سال اول اجرای قرارداد متوقف شد. این در حالی است که طبق تعریف حقوقی، دیوان محاسبات ایران، حق دخالت در قراردادها، پیش از آنکه قراردادها اجرایی و مالی شوند را ندارد. |         طرفین قرار داد کرسنت کیا بودن؟          | کرسنت پترولیوم و شرکت ملی نفت ایران |
| چهارشنبه‌سوری |  چهارشنبه‌سوری یکی از جشن‌های ایرانی است که از غروب آخرین سه‌شنبه ی ماه اسفند، تا پس از نیمه‌شب تا آخرین چهارشنبه ی سال، برگزار می‌شود و برافروختن و پریدن از روی آتش مشخصهٔ اصلی آن است. این جشن، نخستین جشن از مجموعهٔ جشن‌ها و مناسبت‌های نوروزی است که با برافروختن آتش و برخی رفتارهای نمادین دیگر، به‌صورت جمعی در فضای باز برگزار می‌شود. به‌گفتهٔ ابراهیم پورداوود چهارشنبه‌سوری ریشه در گاهنبارِ هَمَسْپَتْمَدَم زرتشتیان و نیز جشن نزول فروهرها دارد که شش روز پیش از فرارسیدن نوروز برگزار می‌شد. احتمال دیگر این است که چهارشنبه‌سوری بازمانده و شکل تحول‌یافته‌ای از جشن سده باشد، که احتمال بعیدی است. علاوه برافروختن آتش، آیین‌های مختلف دیگری نیز در بخش‌های گوناگون ایران در زمان این جشن انجام می‌شوند. برای نمونه، در تبریز، مردم به چهارشنبه‌بازار می‌روند که با چراغ و شمع، به‌طرز زیبایی چراغانی شده‌است. هر خانواده یک آینه، دانه‌های اسفند، و یک کوزه برای سال نو خریداری می‌کنند. همه‌ساله شهروندانی از ایران در اثر انفجارهای ناخوشایند مربوط به این جشن، کشته یا مصدوم می‌شوند.   |         نام جشن اخرین شنبه ی سال چیست؟          |              No Answer              |

### Statistic

| Split | # of instances | # of unanswerables | avg. question length | avg. paragraph length | avg. answer length |
| :---: | :------------: | :----------------: | :------------------: | :-------------------: | :----------------: |
| Train |     9,000      |       2,700        |         8.39         |        224.58         |        9.61        |
| Test  |      938       |        280         |         8.02         |        220.18         |        5.99        |

The lengths are on token level.

To learn more about the data and more example take a look [here](https://github.com/sajjjadayobi/PersianQA/tree/main/dataset#readme).

## Models

Currently, two models (baseline) on
[HuggingFace🤗](https://huggingface.co/SajjadAyoubi/) model hub are using the
dataset. The models are listed in the table below.

|                                          Name                                          | Params |              Training              |
| :------------------------------------------------------------------------------------: | :----: | :--------------------------------: |
| [xlm-roberta-large-fa-qa](https://huggingface.co/SajjadAyoubi/xlm-roberta-large-fa-qa) |  558M  | fine-tuned on SQuAD2.0 + PersianQA |
|         [bert-base-fa-qa](https://huggingface.co/SajjadAyoubi/bert-base-fa-qa)         |  162M  |      fine-tuned on PersianQA       |

You can try out our existing models and study examples. For more information
on the examples, visit [this page](https://huggingface.co/SajjadAyoubi).

**In case you have trained any model on the dataset, we would be more than glad
to hear the details. Please, make a pull request for that regards. A simple
notebook for training baseline can be found
[at here](https://github.com/sajjjadayobi/PersianQA/blob/main/notebooks/TrainBaseline.ipynb).**

### How to use

All the examples are based on the Bert version but you can use other versions as well.

#### Requirements

Transformers require `transformers` and `sentencepiece`, both of which can be
installed using `pip`.

```sh
pip install transformers sentencepiece
```

#### Pipelines 🚀

In case you are not familiar with Transformers, you can use pipelines instead.

Note that, pipelines can't have _no answer_ for the questions.

```python
from transformers import pipeline

model_name = "SajjadAyoubi/bert-base-fa-qa"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

text = "سلام من سجاد ایوبی هستم و به پردازش زبان طبیعی علاقه دارم"
questions = ["اسمم چیه؟", "علاقه مندیم چیه؟"]

for question in questions:
    print(qa_pipeline({"context": text, "question": question}))

# >>> {'score': 0.5183013081550598, 'start': 8, 'end': 18, 'answer': 'سجاد ایوبی'}
# >>> {'score': 0.22757135331630707, 'start': 29, 'end': 46, 'answer': 'پردازش زبان طبیعی'}
```

#### Manual approach 🔥

Using the Manual approach, it is possible to have _no answer_ with even better
performance.

- PyTorch

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from src.utils import AnswerPredictor

model_name = "SajjadAyoubi/bert-base-fa-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

text = "سلام من سجاد ایوبی هستم و به پردازش زبان طبیعی علاقه دارم"
questions = ["اسمم چیه؟", "علاقه مندیم چیه؟", "من چند سالمه؟"]

# this class is from src/utils.py and you can read more about it
predictor = AnswerPredictor(model, tokenizer, device="cpu", no_answer=True)
preds = predictor(questions, [text] * 3, batch_size=3)

for k, v in preds.items():
    print(v)
```

Produces an output such below:

```
100%|██████████| 1/1 [00:00<00:00,  3.56it/s]
{'score': 9.57140064239502, 'text': 'سجاد ایوبی'}
{'score': 9.273895263671875, 'text': 'پردازش زبان طبیعی'}
{'score': 7.138418197631836, 'text': ''}
```

- TensorFlow 2.X

```python
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
from src.utils import TFAnswerPredictor

model_name = "SajjadAyoubi/bert-base-fa-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

text = "سلام من سجاد ایوبی هستم و به پردازش زبان طبیعی علاقه دارم"
questions = ["اسمم چیه؟", "علاقه مندیم چیه؟", "من چند سالمه؟"]

# this class is from src/utils.py, you can read more about it
predictor = TFAnswerPredictor(model, tokenizer, no_answer=True)
preds = predictor(questions, [text] * 3, batch_size=3)

for k, v in preds.items():
    print(v)
```

Produces an output such below:

```text
100%|██████████| 1/1 [00:00<00:00,  3.56it/s]
{'score': 9.57140064239502, 'text': 'سجاد ایوبی'}
{'score': 9.273895263671875, 'text': 'پردازش زبان طبیعی'}
{'score': 7.138418197631836, 'text': ''}
```

Or you can access the whole demonstration using [HowToUse Notebook on
Google
Colab](https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/HowToUse.ipynb)

### Evaluation

To evaluate your models, you can use the provided [evaluation script](https://github.com/sajjjadayobi/PersianQA/blob/main/src/evaluation.py).

Although, the GLEU metrics are not the best measures to evaluate the model on,
the results are as shown below. The best baseline scores bolder.

#### On [ParsiNLU](https://github.com/persiannlp/parsinlu)

[ParsiNLU](https://github.com/persiannlp/parsinlu) is a comprehensive suit of
high-level NLP tasks for Persian language. This suit contains 6 different key
NLP tasks including Question Answering and contains 570 questions (without any
unanswerable questions).

|         Model         | F1 Score  | Exact Match | Params |
| :-------------------: | :-------: | :---------: | :----: |
|         Human         |   86.2%   |      -      |   -    |
| Our XLM-Roberta-Large | **78.6%** |   52.10%    |  558M  |
|     Our ParsBERT      |   62.6%   |   35.43%    |  162M  |
| ParsiNLU's mT5-small  |   28.6%   |      -      |  300M  |
|  ParsiNLU's mT5-base  |   43.0%   |      -      |  582M  |
| ParsiNLU's mT5-large  |   60.1%   |      -      |  1.2B  |
|   ParsiNLU's mT5-XL   |   65.5%   |      -      |   -    |

#### On PersianQA testset

|         Model         |  F1 Score  | Exact Match | Params |
| :-------------------: | :--------: | :---------: | :----: |
| Our XLM-Roberta-Large | **84.81%** |   70.40%    |  558M  |
|     Our ParsBERT      |   70.06%   |   53.55%    |  162M  |

## Experiment: Use Cross Lingual Transfer Learning 

As far as we managed to experiment with the dataset, the best results always
came from merging the dataset with other big datasets (in other languages) such
as SQuAD using multilingual models. Foremost, try to establish the "reading
comprehension" concept in your model with the larger dataset and then transfer
the knowledge to Persian with this very dataset.

However, this method is not only limited to this application and can be put to
use in other domains and smaller datasets.

## Contact us

If you have a technical question regarding the dataset, code or publication,
please create an issue in the repository. This is the fastest way to reach us.

<!-- TODO: we would be happy to hear from you about better models -->

## Citation

As of yet, we didn't publish any papers on the work.
However, if you did, please cite us properly with an entry like one below.

```bibtex
@misc{PersianQA,
  author          = {Ayoubi, Sajjad \& Davoodeh, Mohammad Yasin},
  title           = {PersianQA: a dataset for Persian Question Answering},
  year            = 2021,
  publisher       = {GitHub},
  journal         = {GitHub repository},
  howpublished    = {\url{https://github.com/SajjjadAyobi/PersianQA}},
}
```

## Acknowledgment

The process of bringing this dataset up and providing it, much like any
other work in the field, is a cumbersome and costly task. This is but a tiny
help to Persian Open-Source community and we are sincerely wishing it provides
inspiration and ground work for other Free projects.

- Thanks to [_Navid Kanani_](https://github.com/navidkanaani) and _Abbas Ayoubi_
- Thanks to Google Colab and HuggingFace🤗 for making this work easier

At last, the tool we used for collecting the data, managing and exporting it is
available under a Free and Libre software license at [_thecollector_
repository](https://github.com/Davoodeh/thecollector).
