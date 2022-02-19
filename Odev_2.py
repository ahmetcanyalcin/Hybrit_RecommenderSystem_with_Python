# Hybrid Recommender System
#
# İş Problemi
# ID'si verilen kullanıcı için item-based ve
# user-based recommender yöntemlerini
# kullanarak tahmin yapınız.

# Veri Seti Hikayesi
# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır.
# İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını
# barındırmaktadır.
# 27.278 filmde 2.000.0263 derecelendirme içermektedir.
# Bu veriler 138.493 kullanıcı tarafından
# 09 Ocak 1995 ile 31 Mart 2015
# tarihleri arasında oluşturulmuştur. Bu veri seti ise 17 Ekim 2016 tarihinde
# oluşturulmuştur.
# Kullanıcılar rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy
# verdiği bilgisi mevcuttur.


###############################################################################################

###################################################
# Görev 1: Veri hazırlama işlemleri
#########################################

import pandas as pd

pd.set_option('display.max_columns', 20)

movie = pd.read_csv('Veri Setleri/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Veri Setleri/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

df.shape

df["title"].nunique()  # benzersiz film isimleri

df["title"].value_counts().head()  # her film isminin kaç defa geçtiği

# Her filmin kaç defa yorum aldığına bakıp bunu da yeni bir DF içine yazdırdık.
comment_counts = pd.DataFrame(df["title"].value_counts())
# 1000 defadan az yorum alan filmleri eledik/çıkarttık.
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

common_movies.shape
common_movies["title"].nunique()
# item based hesabını yapabilmek için tabloyu matrix'e dönüştürüyoruz
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape



# seçilen filme göre öneri yapar
movie_name = "101 Dalmatians (1996)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


# # Rastgele seçilen filme göre öneri yapar
# movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
# movie_name = user_movie_df[movie_name]
# user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


# burada ilgili kullanıcının öneri filmlerini pickle metodu ile sakladık
# user_movie_df'in kaydedilmesi
# import pickle
# pickle.dump(user_movie_df, open("user_movie_df.pkl", 'wb'))
#
# # user_movie_df'inin yüklenmesi
# user_movie_df = pickle.load(open('user_movie_df.pkl', 'rb'))

# len(movies_watched) # burada ise 33 tane film izlediğini saydırdık


###################################################
# Görev2: Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz.
#########################################


random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
# yukarıdaki random_state ile 28941 nolu kullanıcı seçildi

# burada bu kullanıcının izlediği filmleri belirliyor
random_user_df = user_movie_df[user_movie_df.index == random_user]
# fakat yukarıdaki df de boş veya NA değerler de var bunları görmeden değerleri çıkartıp bunları listeye atadık

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# aşağıda ise izlediği filmlerden seçip gerçekten de bu filme puan vermiş mi kontrol ettik.!
# user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Schindler's List (1993)"]






#############################################
# Görev 3 : Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

pd.set_option('display.max_columns', 5)

# movies_watched_df oluşturup ardından da fancy index ile buna atama yaptık
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

# user_movie_count ile boş değer bulunmayan kullanıcıların izlediği filmleri topladık
user_movie_count = movies_watched_df.T.notnull().sum()

# reset index  metodu ile kullanıcı kodlarını indexden kurtardık
user_movie_count = user_movie_count.reset_index()
# tablo başlığını ekledik
user_movie_count.columns = ["userId", "movie_count"]

# kullanıcılar arasında 20 den az film izleyen kullanıcıları eledik
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# bizim kullanıcımızda olduğu gibi 33 tane film izleyen var mı buna baktık
user_movie_count[user_movie_count["movie_count"] == 33].count()

# kullanıcmızla aynı filmleri izleyen kullanıcıları atadık ve useid lerini seçtik
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

users_same_movies.count()


#############################################
# Görev 4 : Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz.
#############################################


# concat metodu ile kullanıcının izlediği filmler ile aynı filmleri izleyen kullanıcıları birleştirdik ve belirledik
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      random_user_df[movies_watched]])

final_df.shape

#  corr matrisinin user id değişkenleri için oluşturduk
final_df.T.corr()

# bu corr matrisinden duplicates verileri kaldıkdır ve bunu unstack ile matris yaptık aynı zamanda sıralama yaptık

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
# ilgili corr_df i yeni bir dataframe içine atadık.
corr_df = pd.DataFrame(corr_df, columns=["corr"])
# dataframede sütun isimlerini değiştirdik
corr_df.index.names = ['user_id_1', 'user_id_2']
# buradaki userid1 değerini indexden kurtardık
corr_df = corr_df.reset_index()

# yukarıda index den kurtardığım user idler için:
#
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

# top_userları corr'a göre sıraladık
top_users = top_users.sort_values(by='corr', ascending=False)

# sıraladığım tablonun sütun isimlerini değiştirdik
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)



#############################################
# Görev 5 : Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutunuz.
#############################################
# burada weighted avarage hesaplayabilmek için ilgili kullanıcıların rating değerlerini getirdik
rating = pd.read_csv('Veri Setleri/movie_lens_dataset/rating.csv')

# kullanıcıların rating değerlerini getirip bunlrı merge yaptık 3 değişkene göre
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')


# corr ve rating değerlerini çarparak ağırlıklı ratingleri hesapladık
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# groupby metodu ile movieid lerin ortalama ağırlıklı ratinglerini hesapladık ve recommentatio_df'e atadık.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# ağırlıklı ortalaması 4 den büyük olanlara baktık
recommendation_df[recommendation_df["weighted_rating"] > 4]

# önerilmesi gereken filmleri belirledik
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4]\
    .sort_values("weighted_rating", ascending=False)
# önerilmesi gereken filmlerin isimlerini movies filminden getirdik
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])

movies_to_be_recommend_top5 = movies_to_be_recommend.merge(movie[["movieId", "title"]]).head()

#############################################
# Görev 6 : Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
#############################################
# ▪ 5 öneri user-based
# ▪ 5 öneri item-based
# olacak şekilde 10 öneri yapınız.

