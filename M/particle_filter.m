clear vars;close all;clc;
% get the x y theta range 4 variables
load observation_3.mat;

load ground_truth_3.mat;
error_per_trial = 0
mse_per_trial = 0
for trial = 1:5

plot_impact_traces = 0;
plot_impact_hist = 0;
num_observations = length(x);

capacity = 500; % capacity should larger than or equal with the num_observations
num_particles = 200;

% simple particles consist of x, y, vx, vy, -- gravity is constant (9.8m/s)
% p(x_t | x_t-1) ~ Normal about physics
% p(y_t | x_t) ~ Normal w/ unknown variances std_x and std_y
% p(sigma_*) ~ Gamma(1,50)
% p(v_0) ~ Uniform(100,1000);
% p(x_0) ~Uniform(-100000,100000);

x_t = logical([1 0 0 0 0 0]);
y_t = logical([0 1 0 0 0 0]);
vx_t = logical([0 0 1 0 0 0]);
vy_t = logical([0 0 0 1 0 0]);
std_x = logical([0 0 0 0 1 0]);
std_y = logical([0 0 0 0 0 1]);

particles = zeros(num_particles,6,capacity);
expectation = zeros(6,capacity);

% initialize_particles5
% 3 dimension: 1st -> num_particles 
%                       2nd -> particles self dimension 
%                       3rd -> time index (from 1 to num_observations)
particles(:,x_t,1) = linspace(-10000,10000,num_particles);
particles(:,vx_t,1) = cos(pi/4)*-sign(particles(:,x_t)).*rand(1,num_particles)'*(1000-100)+100;
particles(:,vy_t,1) = sin(pi/4)*rand(1,num_particles)*(1000-100)+100;
particles(:,std_x,1) = sqrt(10000);%gamrnd(1,10,num_particles,1);%
particles(:,std_y,1) = sqrt(500);%gamrnd(1,10,num_particles,1);%

for t = 1:num_observations
    % 1. compute weights from likelihood
    pobsx_given_x = normpdf(particles(:,x_t,t),x(t),particles(:,std_x,t));%.*gampdf(particles(:,std_x,t),1,50);
    pobsy_given_y = normpdf(particles(:,y_t,t),y(t),particles(:,std_y,t));%.*gampdf(particles(:,std_y,t),1,50);

    weights = pobsx_given_x.*pobsy_given_y;
    weights = weights/sum(weights);

    % 2. compute the rao-blackwellized posterior expectation
    expectation(:,t) = weights'*particles(:,:,t);

    % 3. resample particle according to their weights
    [inds] = resample(weights,num_particles);
    particles(:,:,t) = particles(inds,:,t);

    % 4. update particles accordingn to state model
    particles(:,std_x,t+1) =  particles(:,std_x,t);
    particles(:,std_y,t+1) =  particles(:,std_y,t);
    particles(:,x_t,t+1) =  particles(:,x_t,t)+delta_t*particles(:,vx_t,t)+randn(num_particles,1)*20;
    particles(:,y_t,t+1) =  particles(:,y_t,t)+delta_t*particles(:,vy_t,t)+randn(num_particles,1)*20;
    particles(:,vx_t,t+1) =  particles(:,vx_t,t)+randn(num_particles,1)*10;
    particles(:,vy_t,t+1) =  particles(:,vy_t,t)-9.8*delta_t+randn(num_particles,1)*10;


    % plot diagnostic plots
    figure(2)
    if(plot_impact_hist)
        subplot(2,1,1)
    end
    plot(expectation(x_t,1:t),expectation(y_t,1:t))
    hold on
    scatter(particles(:,x_t,t),particles(:,y_t,t),'o')
    %scatter(x(1:t),y(1:t),'bx');
    if plot_impact_traces
        tf = t+1:5:capacity*2;
        tf = tf*delta_t-t*delta_t;

        fx = repmat(particles(:,x_t,t+1),1,length(tf)) + repmat(tf,num_particles,1).*repmat(particles(:,vx_t,t+1),1,length(tf));
        fy = repmat(particles(:,y_t,t+1),1,length(tf)) + repmat(tf,num_particles,1).*repmat(particles(:,vy_t,t+1),1,length(tf)) - repmat((1/2)*9.8*tf.^2,num_particles,1);

        plot(fx',fy','g-')
    end
    plot(true_x,true_y,'r')
    line([-12000 12000],[0 0],'LineWidth',2,'Color',[0 0 0])
    hold off
    set(gca,'XLim',[-12000 12000],'YLim',[-100 750])

    if plot_impact_hist
        subplot(2,1,2)
        touchdown = zeros(1,num_particles)+inf;
        for p = 1:num_particles
            ind = find(fy(p,:)<=0,1);

            if ~isempty(ind)
                touchdown(p) = fx(p,ind);
            end
        end
        edges =-10000:20:10000;
        [N,bin] = histc(touchdown(touchdown ~= inf),edges);
        bh = bar(edges,N,'histc');
        set(gca,'XLim',[-1000 1000])
    end
    drawnow
    pause(.01)


end



eh= plot(expectation(x_t,:),expectation(y_t,:),'r');
hold on
th = plot(true_x,true_y,'g');
hold off
df = expectation(x_t|y_t,1:t)-([true_x; true_y]);
mse_pf = mean(sum(df.^2,1));
error_pf = norm(expectation(x_t|y_t,t)-[true_x(end) true_y(end)]');
legend('Estimate', 'True')
title('Particle filtering')
axis 'tight'

xlabel('X')
ylabel('Y')

error_per_trial = error_per_trial + error_pf;
mse_per_trial = mse_per_trial + mse_pf;

end
text(3000,250,['Ave. MSE: ' num2str(mse_per_trial/5) ', Ave. Last point error: ' num2str(error_per_trial/5)])
