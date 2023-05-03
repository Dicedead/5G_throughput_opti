clear;
clc;
% load real data
load('.\data_2.mat');

% round angles into (0,2pi)
for i = 1 : size(src_model,1)
    if src_model(i,3) < 0
        src_model(i,3) = src_model(i,3) + 2*pi;
    end
end

lambda = 0.336845458426966;  % wave length
N_user = size(src_model,1); % parameters from DOA estimation: # user, # slot
N_slot = size(directions,2);

C0 = 0.8;   % constant representing the gap between perfect channel coding and real applications
noise_sig = 0.1; % white noise variance
B = 2e6 / N_user;    % system bandwidth

p = xyz;  % location of all antennas on the BS
r_user = src_model(:,1:2);  % position of the users

R_user = zeros(N_slot, N_user);  % data rate

for n_slot = 1 : N_slot
    sigma = sqrt(2 - 2 * cos(Theta{n_slot} / 360 * 2 * pi)); % variance of the Gaussian distribution
    w = zeros(size(p,1), size(sigma,2));
    for n_cluster = 1 : size(sigma,2)   % multiple main lobes
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%% flexibeam weights %%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        r0_angle = directions{1, n_slot}(n_cluster) / 360 * 2 * pi;  % the target direction in this cluster
        r0 = [cos(r0_angle) sin(r0_angle)];
        
        w_ampl = zeros(size(p,1), 1);  % amplitude of beamforming weights
        for i = 1 : size(p,1)
            w_ampl(i) = exp(-2 * (pi^2) * (sigma(n_cluster)^2) * (norm(p(i,:))^2) / (lambda^2));
            w(i, n_cluster) = w_ampl(i) * exp(-1i * 2*pi * dot(p(i,:), r0) / lambda);
        end
        
        % do normalization
        sum_w_norm = 0;
        for i = 1 : size(p,1)
            sum_w_norm = sum_w_norm + abs(w(i,n_cluster))^2;
        end
        w(:, n_cluster) = w(:, n_cluster) / sqrt(sum_w_norm);
        
    end
    % do normalization on all weights
    w_sum = sum(w,2);
    w_sum = w_sum / sqrt(sum(abs(w_sum).^2,'all'));
    
    b = zeros(N_user, 1); % beam shape
    b_gain = zeros(N_user, 1);  % gain of each antenna
    for j = 1 : N_user
        for i = 1 : size(p,1)
            b(j) = b(j) + w_sum(i) * exp(1i * 2*pi * dot(p(i,:), r_user(j,:)) / lambda);
        end
        b_gain(j) = abs(b(j));
        R_user(n_slot,j) = B * log2(1 + ((b_gain(j))^2 / (2 * noise_sig)) * C0);
    end
    
end

R_aver = mean(R_user,'all');
