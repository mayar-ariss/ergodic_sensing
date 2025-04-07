clear; clc; close all;

% Parameters
N_values = [100, 200, 300, 400];   % Different sample sizes
num_trials = 30;                   % Number of independent trials
epsilon = 0.005;                   % Convergence precision threshold

% Define sensing kernel
phi = @(x, y) x .* y;
theoretical_mean = 1/4;

% Prepare figure for subplots
figure('Color', 'w', 'Position', [100, 100, 1200, 800]);

% Print header in command window
fprintf('\n--- Convergence Thresholds (k*) for Each Sample Size ---\n');

% Loop through sample sizes
for idx = 1:length(N_values)
    N = N_values(idx);

    % Generate uniform random samples
    x = rand(num_trials, N);
    y = rand(num_trials, N);

    % Compute cumulative empirical means
    empirical_means = zeros(num_trials, N);
    for t = 1:num_trials
        samples = phi(x(t,:), y(t,:));
        empirical_means(t, :) = cumsum(samples) ./ (1:N);
    end

    % Compute mean and standard error across trials
    mean_curve = mean(empirical_means, 1);
    stderr = std(empirical_means, 0, 1) / sqrt(num_trials);
    k_star = find(stderr < epsilon, 1, 'first');

    % Output threshold value
    if ~isempty(k_star)
        fprintf('N = %3d  →  k* = %3d (stderr < %.4f)\n', N, k_star, epsilon);
    else
        fprintf('N = %3d  →  k* not reached (stderr > %.4f)\n', N, epsilon);
    end

    % Subplot
    subplot(2, 2, idx); hold on;

    % Plot individual trials (light gray)
    h_trials = plot(1:N, empirical_means', '-', 'Color', [0.6 0.6 0.6 0.5]);

    % Confidence band
    fill([1:N, fliplr(1:N)], ...
         [mean_curve + stderr, fliplr(mean_curve - stderr)], ...
         [0.4 0.4 1], 'EdgeColor', 'b', 'FaceAlpha', 0.3);

    % Mean curve
    h_mean = plot(1:N, mean_curve, 'b-', 'LineWidth', 1);

    % Theoretical mean line
    h_theory = yline(theoretical_mean, 'r--', 'LineWidth', 1);

    % Mark k* on curve
    if ~isempty(k_star)
        h_thresh = plot(k_star, mean_curve(k_star), 'ko', 'MarkerFaceColor', 'r');
        threshold_str = sprintf('k^* = %d', k_star);
    else
        h_thresh = NaN;
        threshold_str = 'Threshold not reached';
    end

    % Labels and title
    xlabel('Number of Samples (k)');
    ylabel('Empirical Mean of \phi(x, y)');
    title(sprintf('N = %d (%s)', N, threshold_str), 'FontWeight', 'bold');
    grid on;

    % Add legend to each subplot
    legend([h_trials(1), h_mean, h_theory, h_thresh], ...
        {'Empirical Realizations', 'Mean Across Trials', ...
         'Theoretical Integral', threshold_str}, ...
         'Location', 'northeast', 'FontSize', 7);
end

% Global title
sgtitle('\bfConvergence of Empirical Means to Theoretical Value (\phi(x, y) = x \cdot y)', 'FontSize', 14);
