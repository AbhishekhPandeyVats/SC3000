% Facts: Define the birth order of Queen Elizabeth's offsprings
birth_order(prince_charles, 1).
birth_order(princess_ann, 2).
birth_order(prince_andrew, 3).
birth_order(prince_edward, 4).

% Rule: Define the old Royal succession rule
old_succession(LineOfSuccession) :-
    findall(X, (birth_order(X, _), male(X)), MaleList), % Find all males in birth order
    findall(X, (birth_order(X, _), female(X)), FemaleList), % Find all females in birth order
    append(MaleList, FemaleList, LineOfSuccession). % Concatenate males and females for the line of succession

% Helper rule: Define whether a person is male
male(prince_charles).
male(prince_andrew).
male(prince_edward).

% Helper rule: Define whether a person is female
female(princess_ann).
