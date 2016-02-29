% Matthew Aitken
% HMM model dynamics

clear all
close all
clc


%%
% States (Y Branch problem) 1-7
d = 0.5;
state_xpos = [0,0,0,d*cosd(135),d*2*cosd(135),d*cosd(45),d*2*cosd(45)];
state_ypos = [0,d,d*2,d*2+d*sind(135),d*2+d*2*sind(135),d*2+d*sind(45),d*2+d*2*sind(45)];

figure(1)
hold on
plot(state_xpos,state_ypos,'Ob')
branch = [3 6 7];
plot(state_xpos(1:5),state_ypos(1:5),'k')
plot(state_xpos(branch),state_ypos(branch),'k')

for i=1:length(state_xpos)
    text(state_xpos(i)+0.05,state_ypos(i),sprintf('X_%d', i-1))
end
text(state_xpos(1)+0.05,state_ypos(1)-.1,'Start')
text(state_xpos(5)-0.05,state_ypos(5)-.1,'Exit 1')
text(state_xpos(7)+0.05,state_ypos(7)-.1,'Exit 2')
text(state_xpos(4)-0.075,state_ypos(4)-.1,'UGS 1')
text(state_xpos(6)+0.05,state_ypos(6)-.1,'UGS 2')
hold off


%% 
% Transition probability
rt = 0.3; %Probability of leaving a state (used for real model)
rT = [1-rt,rt,0,0,0,0,0;
     rt/2,1-rt,rt/2,0,0,0,0;
     0,rt/3,1-rt,rt/3,0,rt/3,0;
     0,0,rt/2,1-rt,rt/2,0,0;
     0,0,0,rt,1-rt,0,0;
     0,0,rt/2,0,0,1-rt,rt/2;
     0,0,0,0,0,rt,1-rt];
rT = rT';

% t = 0.2; % Probability of transitioning between states
% T = [1-t,t,0,0,0,0,0;
%      t,1-2*t,t,0,0,0,0;
%      0,t,1-3*t,t,0,t,0;
%      0,0,t,1-2*t,t,0,0;
%      0,0,0,t,1-t,0,0;
%      0,0,t,0,0,1-2*t,t;
%      0,0,0,0,0,t,1-t];
 
t = 0.2; % Probability of moving towards state 1
T = [1,0,0,0,0,0,0;
     t,1-t,0,0,0,0,0;
     0,t,1-t,0,0,0,0;
     0,0,t,1-t,0,0,0;
     0,0,0,t,1-t,0,0;
     0,0,t,0,0,1-t,0;
     0,0,0,0,0,t,1-t];
     
T = T';

% Observation Probabilities
a_i = .85; % chance of a true detection
b_i = .05; % chance of a false detection

B = [b_i,1-b_i,b_i,1-b_i; %cols: 4 detect/no, 6 detect/no,
     b_i,1-b_i,b_i,1-b_i; %rows: states 
     b_i,1-b_i,b_i,1-b_i;
     a_i,1-a_i,b_i,1-b_i;
     b_i,1-b_i,b_i,1-b_i;
     b_i,1-b_i,a_i,1-a_i;
     b_i,1-b_i,b_i,1-b_i];

% Initial belief
p_x_initial = [0 0 0 0 0.5 0 0.5]';
initial_x = randsample(7,1,true,p_x_initial);

figure(2)
for i=1:30
    % Update (prior) belief and true position
    if i == 1
        p_x_new = p_x_initial;
        x_new = initial_x;
    else
        p_x_new = T*p_x_old;
        x_new = randsample(7,1,true,rT(:,x_old)); % doesn't match model
%         x_new = randsample(7,1,true,T(:,x_old)); % matches model
    end
    
    display(x_new)
    
    % check detections and update the map
    if x_new == 4 
        detect4 = randsample(2,1,true,[1-a_i, a_i])-1;
    elseif x_new == 6
        detect6 = randsample(2,1,true,[1-a_i, a_i])-1;
    else
        detect4 = randsample(2,1,true,[1-b_i, b_i])-1;
        detect6 = randsample(2,1,true,[1-b_i, b_i])-1;
    end
    figure(1)
    hold on
    if detect4 == 1
        plot(state_xpos(4),state_ypos(4),'Or')
        py4 = diag(B(:,1));
        display('UGS4 = true')
    else 
        plot(state_xpos(4),state_ypos(4),'Ob')
        py4 = diag(B(:,2));
        display('UGS4 = false')
    end
    if detect6 == 1
        plot(state_xpos(6),state_ypos(6),'Or')
        py6 = diag(B(:,3));
        display('UGS6 = true')
    else
        plot(state_xpos(6),state_ypos(6),'Ob')    
        py6 = diag(B(:,4));
        display('UGS6 = false')
    end
    if i ~= 1
        delete(pos)
    end
    pos = plot(state_xpos(x_new),state_ypos(x_new),'Xg','MarkerSize',15);    
    hold off
    
    % Fuse detections (conditionally independant)
    p_x_new = py4*py6*p_x_new;
    p_x_new = p_x_new/norm(p_x_new,1);
    
    entropy = 0;
    for j=1:length(p_x_new)
        if p_x_new(j) ~= 0
            entropy = entropy - p_x_new(j)*log(p_x_new(j));
        end
    end
    entropy = entropy
    
    % Update probability belief
    figure(2)
    bar(p_x_new)
    ylim([0 1])
    title('Probability of the Pursuer''s position')
    xlabel('Node')
    ylabel('P(y)')
    
%     display(i)
%     display(sum(p_x_new(1:3)))
    
    p_x_old = p_x_new;
    x_old = x_new;   
    
    pause
    if i == 1
        display('Press enter to step')
    end
%     pause(0.5)
end

