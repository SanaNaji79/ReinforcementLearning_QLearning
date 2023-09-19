%% README
%in this code we try to implement a learning approach for an agent to learn
%a map with a goal and hell position starting from a random point. this
%goal is reached using delta-rule having different learning rates and
%discount factors.
%%
clear all ;
clc ;
%% initialization
map = zeros(15 , 15) ;
target = [4 , 3] ;
map(target(1) , target(2)) = 10 ;
cat = [10 , 11] ;
map(cat(1) , cat(2)) = 5 ;
Q_value = zeros(15 , 15 , 4) ;
Q_value(target(1) , target(2) , :) = 0.5 ;
Q_value(cat(1) , cat(2) , :) = 0.5 ;
LR = 0.8 ; 
DF = 0.8 ;
%%
posible_move = cell(15 , 15) ;
for i = 1:15
    for j = 1:15
        if i == 15
            if j == 15
                posible_move{i , j} = [1 , 4] ;
            elseif j == 1
                posible_move{i , j} = [1 , 2] ;
            else
                posible_move{i , j} = [1 , 2 , 4] ;
            end
        elseif i == 1
            if j == 15
                posible_move{i , j} = [3 , 4] ;
            elseif j == 1
                posible_move{i , j} = [2 , 3] ;
            else
                posible_move{i , j} = [2 , 3 , 4] ;
            end 
        else
            if j == 15
                posible_move{i , j} = [1 , 3 , 4] ;
            elseif j == 1
                posible_move{i , j} = [1 , 2 , 3] ;
            else
                posible_move{i , j} = [1 , 2 , 3 , 4] ;
            end 
        end
    end
end
%% 
for n = 1:100000
    i = 0 ; 
    while i == 0
        a = randi(15) ;
        b = randi(15) ;
        if ~isequal([a , b] , target)
            if ~isequal([a , b] , cat)
                i = 1 ;
            end
        end
    end
    start = [a , b] ;
    r = map(start(1) , start(2)) ;
    all_moves = [] ;
    locs = start ;
    while r == 0
        moves = posible_move{start(1) , start(2)} ;
        Q = Q_value(start(1) , start(2) , moves) ;
        Q = reshape(Q , [length(Q) , 1]) ;
        max1 = max(Q) ;
        loc = find(Q == max1) ;
        a = randi(length(loc)) ;
        current_move = moves(loc(a)) ;
%         p = (0.1/max([1 , length(Q)-length(loc)]))*ones(length(Q) , 1) ;
%         p(loc) = (1 - 0.1*min([1 , length(Q)-length(loc)]))/length(loc) ;
%         current_move = moves(find(rand<cumsum(p),1,'first')) ;
        next = next_step(start , current_move) ;
        r = map(next(1) , next(2)) ;
        start = next ;
        all_moves = [all_moves , current_move] ;
        locs = [locs ; start] ;
    end
    Q_value = update(Q_value , map , all_moves , locs , LR , DF) ;
end
%%
start1 = [] ;
for i = 1:10
    a = random(target, cat) ;
    start1 = [start1 ; a] ;
end

for i = 1:10
    [locs , all_moves] = find_path(Q_value , posible_move , map , start1(i , :)) ;
    subplot(2 , 5 , i) ;
    plot_path(locs , target , cat) ;
    %title('the path after training') ;
end

%%
Q_value1 = 0.*Q_value ;
Q_value1(target(1) , target(2) , :) = 0.5 ;
Q_value1(cat(1) , cat(2) , :) = -0.5 ;
figure
for i = 1:6
[locs , all_moves] = find_path(Q_value1 , posible_move , map , start) ;
subplot(2 , 3 , i) ;
plot_path(locs , target , cat) ;
title('the path before training') ;
end
%%
figure ;
surf(1:15 , 1:15 , Q_value(: , : , 1)  , 'FaceAlpha' , 0.5) ;
title('horizontal movement to the left') ;
figure ;
surf(1:15 , 1:15 , Q_value(: , : , 2)  , 'FaceAlpha' , 0.5) ;
title('upward movement') ;
figure ;
surf(1:15 , 1:15 , Q_value(: , : , 3)  , 'FaceAlpha' , 0.5) ;
title('horizontal movement to the right') ;
figure ;
surf(1:15 , 1:15 , Q_value(: , : , 4)  , 'FaceAlpha' , 0.5) ;
title('downward movement') ;
%% function declaration
function [next] = next_step(start , move)
next = start ;
if move == 1
    next(1) = next(1) - 1 ;
elseif move == 2
    next(2) = next(2) + 1 ;
elseif move == 3
    next(1) = next(1) + 1 ;
else
    next(2) = next(2) - 1 ;
end
end
function [] = plot_path(locs , target , cat)
for i = 1:length(locs)-1
    plot([locs(i , 1) , locs(i+1 , 1)] , [locs(i , 2) , locs(i+1 , 2)] , 'g') ;
    hold on ;
end
c1 = [0 , 0 , 1] ;
c2 = [0 , 0 , 0] ;
c3 = [1 , 1 , 0] ;
scatter([target(1) , cat(1) , locs(1 , 1)] , [target(2) , cat(2) , locs(1 , 2)] , 25 , [c2 ; c1 ; c3] , 'filled') ;
xlim([0 , 16]) ;
ylim([0 , 16]) ;
end
function [Q_value1] = update(Q_value , map , all_moves , locs , LR , DF)
Q_value1 = Q_value ;
Q_value1(locs(end-1 , 1) , locs(end-1 , 2) , all_moves(end)) = Q_value(locs(end-1 , 1) , locs(end-1 , 2) , all_moves(end)) + LR*(map(locs(end , 1) , locs(end , 2)) + DF*Q_value(locs(end , 1) , locs(end , 2) , all_moves(end))-Q_value(locs(end-1 , 1) , locs(end-1 , 2) , all_moves(end))) ;
if length(locs)-2>0
    for i = (length(locs)-2:-1:1)
        Q_value1(locs(i , 1) , locs(i , 2) , all_moves(i)) = Q_value(locs(i , 1) , locs(i , 2) , all_moves(i))+LR*(DF*Q_value(locs(i+1 , 1) , locs(i+1 , 2) , all_moves(i+1))-Q_value(locs(i , 1) , locs(i , 2) , all_moves(i))) ;
    end
end
end
function [start] = random(target, cat) 
i = 0 ; 
while i == 0
    a = randi(15) ;
    b = randi(15) ;
    if ~isequal([a , b] , target)
        if ~isequal([a , b] , cat)
            i = 1 ;
        end
    end
end
start = [a , b] ;
end
function [locs , all_moves] = find_path(Q_value , posible_move , map , start)
r = map(start(1) , start(2)) ;
all_moves = [] ;
locs = start ;
while r == 0
    moves = posible_move{start(1) , start(2)} ;
    Q = Q_value(start(1) , start(2) , moves) ;
    Q = reshape(Q , [length(Q) , 1]) ;
    max1 = max(Q) ;
    loc = find(Q == max1) ;
    a = randi(length(loc)) ;
    current_move = moves(loc(a)) ;
    next = next_step(start , current_move) ;
    r = map(next(1) , next(2)) ;
    start = next ;
    all_moves = [all_moves , current_move] ;
    locs = [locs ; start] ;
end
end