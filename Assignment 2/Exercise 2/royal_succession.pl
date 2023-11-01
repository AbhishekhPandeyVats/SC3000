% Define the relations
male(prince_charles).
male(prince_andrew).
male(prince_edward).
female(princess_ann).
female(queen_elizabeth).

% Define the parent-child relationships
parent(queen_elizabeth, prince_charles).
parent(queen_elizabeth, princess_ann).
parent(queen_elizabeth, prince_andrew).
parent(queen_elizabeth, prince_edward).

% Define the old Royal succession rule
successor(X, Y) :-
    male(X),                % X is male
    parent(Z, X),           % Z is a parent of X
    parent(Z, Y),           % Z is a parent of Y
    X \= Y,                 % X is not the same as Y
    Y \= princess_ann.      % Y is not princess_ann

% Define the first in line of succession
successor_line(X) :-
    parent(queen_elizabeth, X),
    X \= princess_ann.
