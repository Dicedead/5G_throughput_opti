clear;
clc;

load('.\data_1.mat');   % load real data

% round angles into (0,2pi)
for i = 1 : size(src_model,1)
    if src_model(i,3) < 0
        src_model(i,3) = src_model(i,3) + 2*pi;
    end
end

N_user = size(src_model,1);  % parameters from DOA estimation: # user, # slot, # cluster of users
N_slot = size(directions,1);
N_cluster = size(directions,2);
R_user = zeros(N_slot, N_cluster, N_user);  % data rate for users via TDMA

lambda = 0.336845458426966;  % wave length
sigma = 0.1;   % variance of white noise
C0 = 0.8;   % constant representing the gap between perfect channel coding and real applications
B = 2e6 / N_user;    % bandwidth PER user


p = xyz;  % location of all antennas on the BS
r_user = src_model(:,1:2);  % position of the users

for n_slot = 1 : N_slot  % slot
    for n_cluster = 1 : N_cluster  % cluster
        r0_angle = directions(n_slot,n_cluster) / 360 * 2 * pi;  % the target direction in current beamforming
        r0 = [cos(r0_angle) sin(r0_angle)];
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%% matched beamforming weights %%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        w = zeros(size(p,1), 1);
        for i = 1 : size(p,1)
            w(i) =  exp(-1i * 2*pi * dot(p(i,:), r0) / lambda);
        end
        w = w / sqrt(sum(abs(w).^2,'all'));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%% beam shape for matched beamforming %%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        b = zeros(N_user, 1); % beam shape
        b_gain = zeros(N_user, 1);  % gain of each antenna
        for j = 1 : N_user
            for i = 1 : size(p,1)
                b(j) = b(j) + w(i) * exp(1i * 2*pi * dot(p(i,:), r_user(j,:)) / lambda);
            end
            b_gain(j) = abs(b(j));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% calculate data rate by AWGN channel %%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i = 1 : N_user
            R_user(n_slot, n_cluster, i) = B * log2(1 + ((b_gain(i))^2 / (2 * sigma)) * C0);
        end
        
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %%%%%%%%%%% radiation pattern in (0,2pi) %%%%%%%%%%%%
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         N = 10000;  % 10000 sample on the circle
%         r = zeros(N, 2); 
%         for i = 1 : N
%             theta_i = i / N * 2 * pi;
%             r(i,:) = [cos(theta_i) sin(theta_i)];
%         end
% 
%         b = zeros(N, 1); % beam shape
%         b_gain = zeros(N, 1);  % gain of each antenna
%         for j = 1 : N
%             for i = 1 : 19
%                 b(j) = b(j) + w(i) * exp(1i * 2*pi * dot(p(i,:), r(j,:)) / (lambda));
%             end
%             b_gain(j) = abs(b(j));
%         end
%         polarplot([1:1:N] / N * 2*pi,b_gain)
        
    end
end

% average over all users, all slots and all clusters (TDMA)
R_aver = mean(R_user,'all');
