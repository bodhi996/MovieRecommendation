# Movie Recommender

## Domain 

The project is a content based machine learning movie recommender system that uses the attributes and features of the movie in concern and finds the one closes to it. 

## Idea

Recommender System is a system that seeks to predict or filter preferences according to the user’s choices. Recommender systems are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general. 
Recommender systems produce a list of recommendations in any of the two ways – 
*  Collaborative filtering: Collaborative filtering approaches build a model from the user’s past behavior (i.e. items purchased or searched by the user) as well as similar decisions made by other users. This model is then used to predict items (or ratings for items) that users may have an interest in.
*  Content-based filtering: Content-based filtering approaches uses a series of discrete characteristics of an item in order to recommend additional items with similar properties. Content-based filtering methods are totally based on a description of the item and a profile of the user’s preferences. It recommends items based on the user’s past preferences.

# Achievements

Our application boasts of several features:
1. The application differentiates between verified and non-verified users. The algorithm will give verification symbols to those who have been consistent users of this platform.
2. The app also uses third party features such as Linkedin to give out users better depth into their prospective employees.
3. The app provides for a basket so that the user can select multiple people at one time without having to go through the same process recursively. 
4.  The app also provides for payment gateway so that the user doesn't have to deal with the hassle of negotiation and engage with third party payments procedure. 
5.  The user also has the feature of deleting any of this selection from the Cart/Basket page if he/she reconsiders their decesion. 

## Description of the codebase

 
<img src="Screenshot%20(177).png" width="250" height="250"> <img src="Screenshot%20(178).png" width="250" height="250">

       
The code is well arranged into sub-directories which are -
1. The first page is the ```Frontend``` which comes up once you run the code.
2. The second page is the ```Recommendation``` the algorithm makes for a given movie. 

# How to Run

1. Make sure VS Code is installed in your system.
2. Fork this repository.
3. Clone the forked repository:
~~~
git clone https://github.com/<your github username>/EEESoc
~~~
4. Add a remote to the upstream repository:
~~~
# typing the command below should show you only 1 remote named origin with the URL of your forked repository
git remote -v
# adding a remote for the upstream repository
git remote add upstream https://github.com/bodhi996/EEESoc.github
~~~
5. Run the application.
6. Create a new issue if you face any difficulties (after browsing through StackOverflow on your own) and someone will help you 😁
