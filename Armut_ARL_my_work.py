############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# 1. Veri Ön İşleme
# 2. Birliktelik Kuralları Üretme ve Öneride Bulunma
#
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# GÖREV 1. Veri Ön İşleme
############################################

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)

from mlxtend.frequent_patterns import apriori, association_rules  #!pip install mlxtend indirin

# Adım 1: armut_data.csv dosyasını okutunuz.

df_ = pd.read_csv("ArmutARL-221114-234936/armut_data.csv")
df = df_.copy()
df.head()
df.info()
df.isnull().sum()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir. ServiceID ve
# CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df.head()

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı
# (fatura vb. ) bulunmamaktadır. Association Rule Learning uygulayabilmek için bir sepet (fatura vb.)
# tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti; 2017’in 10.ayında
# aldığı 9_4, 38_4 hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması
# gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.

df["CreateDate"].dtype # dtype('O') değişkenimin tipi Object, bunu sadece ay ve yıl olarak date tipinde almak için;

df["CreateDate"] = pd.to_datetime(df["CreateDate"])

df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")   # strftime ile sadece yıl ve ay kısmını filtreliyoruz.
# Dikkat = Y büyük fakat m küçük olacak. ya da şöyle yapılabilir

# df['NEW_DATE'] = pd.to_datetime(df["CreateDate"]).dt.to_period('M')

df.head()

df["Sepet_Id"] = df["UserId"].astype(str) + "_" + df["New_Date"].astype(str)

df.head()

############################################
# GÖREV 2. Birliktelik Kuralları Üretme ve Öneride Bulunma
############################################

# Adım 1: ARL veri yapısınını hazırlayalım (Invoice-Product Matrix)

# Bunun için GroupBy kullalacağım, ARL veri yapısı için indexte Fatura,Sepet,Sepetteki ürünler gibi bilgilerin yer
# alması gerekir.

arl_matrix = df.groupby(['Sepet_Id', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
type(arl_matrix)
arl_matrix.shape
arl_matrix.dtypes
arl_matrix.columns
arl_matrix.value_counts()
arl_matrix.index


### Pivot Table ile yapılmak istenirse;

arl_matrix_pv = df.pivot_table(index="Sepet_Id", columns="Hizmet", values={"CategoryId": "count"})
arl_matrix_pv = arl_matrix.fillna(0).applymap(lambda x: 1 if x > 0 else 0)
# arl_matrix_pv.columns = arl_matrix_pv.columns.droplevel()
type(arl_matrix_pv)
arl_matrix_pv.dtypes
arl_matrix_pv.columns
arl_matrix_pv.value_counts()
arl_matrix_pv.values
arl_matrix_pv.index



# Not bu iki işlem de tamamen aynı sonucu verir, fakat işlemlerde sorun yaşamamak adına GroupBy'ı kullanıyorum.
# Bazı veri setlerinde Pivot Table kullanmak daha mantıklı olabilir

# Adım 2: Birliktelik kurallarını oluşturalım.

support_itemsets = apriori(arl_matrix, min_support=0.01, use_colnames=True)    # burada tüm ürünlerin support değerlerini çıkardık.

# Şimdi birliktelik kuralını oluşturalım, ve rules isimli df e atalım

rules = association_rules(support_itemsets, metric="support", min_threshold=0.01)  # burada da ARL için ürünlerin;
# antecedents, consequents, support, confidence ve lift derğerleri vardır.
rules.head()
rules.shape
# Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # kuralları lifte göre büyükten kücüğe sıralar. (en uyumlu ilk ürünü yakalayabilmek için)
    # confidence'e göre de sıralanabilir insiyatife baglıdır.
    recommendation_list = [] # tavsiye edilecek ürünler için bos bir liste olusturuyoruz.
    # antecedents: X
    #items denildigi için frozenset olarak getirir. index ve hizmeti birleştirir.
    # i: index
    # product: X yani öneri isteyen hizmet
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product): # hizmetlerde(product) gez:
            if j == product_id:# eger tavsiye istenen ürün yakalanırsa
                recommendation_list.append(list(sorted_rules.loc[i]["consequents"]))
                # index bilgisini i ile tutuyordun bu index bilgisindeki consequents(Y) değerini recommendation_list'e ekle.

    # tavsiye listesinde tekrarlamayı önlemek için:
    # mesela 2'li 3'lü kombinasyonlarda aynı ürün tekrar düşmüş olabilir listeye gibi;
    # set yapısının unique özelliginden yararlanıyoruz.
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count] # :rec_count istenen sayıya kadar tavsiye ürün getir.

arl_recommender(rules,"2_0", 5)