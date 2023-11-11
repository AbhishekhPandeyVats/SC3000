female(elizabeth).
female(ann).
male(charles).
male(andrew).
male(edward).

offspring(charles, elizabeth).
offspring(andrew, elizabeth).
offspring(ann, elizabeth).
offspring(edward, elizabeth).

older(charles, ann).
older(ann, andrew).
older(andrew, edward).

is_older(X,Y):- older(X,Y).
is_older(X,Y):- older(X,Z), is_older(Z,Y).

precedes(X,Y):- male(X),male(Y),is_older(X,Y).
precedes(X,Y):- male(X),female(Y),Y\=elizabeth.
precedes(X,Y):- female(X),female(Y),is_older(X,Y).

insert(A,[B|C],[B|D]):- not(is_older(A,B)),!,insert(A,C,D).
insert(A,C,[A|C]).
succession_sort([A|B],SortList):- succession_sort(B,Tail),insert(A,Tail,SortList).
succession_sort([],[]).
successionList(X,SuccessionList):- findall(Y,offspring(Y, X),ChildNodes), succession_sort(ChildNodes, SuccessionList).