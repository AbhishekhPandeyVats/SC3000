% Facts: Define the birth order of Queen Elizabeth's offsprings
birth_order(prince_charles, 1).
birth_order(princess_ann, 2).
birth_order(prince_andrew, 3).
birth_order(prince_edward, 4).

% Rule: Define the new Royal succession rule
new_succession(LineOfSuccession) :-
    findall(X, birth_order(X, _), LineOfSuccession). % Consider the birth order directly

