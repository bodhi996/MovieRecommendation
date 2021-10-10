from tkinter import *
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

class MyWindow:
    def __init__(self, win):
# Define label/entry/listbox/button
        self.lbl0=Label(win, text='Movie Recomendation System', borderwidth=3, relief="solid", padx=5, pady=10).pack(padx=5, pady=10)
        self.lbl1=Label(win, text='Enter movie name ')
        self.lbl3=Label(win, text='Recomended Movies')
        self.t1=Entry(bd=3)
        self.listbox = Listbox(win, width=30, height=10)  
        self.b1=Button(win, text='Submit', command=self.recomend)
       
# Define button/label placement
        self.lbl1.place(x=75, y=75)
        self.t1.place(x=200, y=75)
        self.b1.place(x=200, y=125)
        self.lbl3.place(x=75, y=170)
        self.listbox.place(x=225, y=170)
       
# Define action
    def recomend(self):
# Get input movie name & Clear list box to display output
        str1=str(self.t1.get())  
        self.listbox.delete(0,'end')        
        movies = pd.read_csv("C:/Users/BODHISATWA/Desktop/Movie rating/movies.csv")
        ratings = pd.read_csv("C:/Users/BODHISATWA/Desktop/Movie rating/ratings.csv")
        final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
        final_dataset.head()
        final_dataset.fillna(0,inplace=True)
        final_dataset.head()
        no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
        no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
        f,ax = plt.subplots(1,1,figsize=(16,4))
        # ratings['rating'].plot(kind='hist')
        plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
        plt.axhline(y=10,color='r')
        plt.xlabel('MovieId')
        plt.ylabel('No. of users voted')
        plt.show()
        f,ax = plt.subplots(1,1,figsize=(16,4))
        plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
        plt.axhline(y=50,color='r')
        plt.xlabel('UserId')
        plt.ylabel('No. of votes by user')
        plt.show()
        final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
        final_dataset
        sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
        sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
        print(sparsity)
        csr_data = csr_matrix(final_dataset.values)
        final_dataset.reset_index(inplace=True)
        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        knn.fit(csr_data)
        def get_movie_recommendation(movie_name):
            n_movies_to_reccomend = 10
            movie_list = movies[movies['title'].str.contains(movie_name)]  
            if len(movie_list):        
                movie_idx= movie_list.iloc[0]['movieId']
                movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
                distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
                rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
                recommend_frame = []
                df = pd.DataFrame()
                for val in rec_movie_indices:
                    movie_idx = final_dataset.iloc[val[0]]['movieId']
                    idx = movies[movies['movieId'] == movie_idx].index
                    recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
                df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
# Populating list box with 10 recomended movie name.
                self.listbox.insert(1, str(df.Title[1]))
                self.listbox.insert(2, str(df.Title[2]))
                self.listbox.insert(3, str(df.Title[3]))
                self.listbox.insert(4, str(df.Title[4]))
                self.listbox.insert(5, str(df.Title[5]))
                self.listbox.insert(6, str(df.Title[6]))
                self.listbox.insert(7, str(df.Title[7]))
                self.listbox.insert(8, str(df.Title[8]))
                self.listbox.insert(9, str(df.Title[9]))
                self.listbox.insert(10, str(df.Title[10]))
                return df
            else:
# Populating list box with message if no recomendation found.
                self.listbox.insert(1, "No movies found.")
                self.listbox.insert(2, "Please check your input.")
                return "No movies found. Please check your input"
        ans=get_movie_recommendation(str1)
        print(ans)

window=Tk()
mywin=MyWindow(window)
window.title('Movie Recomendation System')
window.geometry("450x350+10+10")
window.mainloop()
