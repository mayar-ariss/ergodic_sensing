clear; clc; close all;

% Parameters
N_values = [100, 200, 300, 400];   % Different total sample sizes
num_trials = 30;
epsilon = 0.005;                      % Convergence precision
phi = @(x, y) x .* y;                 % Sensing kernel
f_xy = @(x, y) 36 .* x .* (1 - x) .* y .* (1 - y);  % Bettstetter-like PDF
sample_rwp = @(n) rejection_sample(n, f_xy, [0 1], [0 1]); % Sampler

% Theoretical mean under f_xy (discretized integral)
[Xg, Yg] = meshgrid(linspace(0,1,200), linspace(0,1,200));
pdf_vals = f_xy(Xg, Yg); pdf_vals = pdf_vals / sum(pdf_vals(:));
phi_vals = phi(Xg, Yg);
theoretical_mean = sum(phi_vals(:) .* pdf_vals(:));

% Prepare figure
figure('Color', 'w', 'Position', [100, 100, 1200, 800]);

% Print header in command window
fprintf('\n--- Convergence Thresholds (k*) under RWP-inspired PDF ---\n');

% Loop through each N and subplot
for idx = 1:length(N_values)
    N = N_values(idx);
    empirical_means = zeros(num_trials, N);

    for t = 1:num_trials
        [x, y] = sample_rwp(N);
        samples = phi(x, y);
        empirical_means(t, :) = cumsum(samples)' ./ (1:N);
    end

    % Compute statistics
    mean_curve = mean(empirical_means, 1);
    stderr = std(empirical_means, 0, 1) / sqrt(num_trials);
    k_star = find(stderr < epsilon, 1, 'first');
    if isempty(k_star), k_star = NaN; end

    % Output to console
    if ~isnan(k_star)
        fprintf('N = %3d  →  k* = %3d (stderr < %.4f)\n', N, k_star, epsilon);
    else
        fprintf('N = %3d  →  k* not reached (stderr > %.4f)\n', N, epsilon);
    end

    % Subplot
    subplot(2, 2, idx); hold on;

    % Plot all trial curves (light gray)
    h_trials = plot(1:N, empirical_means', '-', 'Color', [0.6 0.6 0.6 0.5]);

    % Shaded confidence band
    fill([1:N, fliplr(1:N)], ...
         [mean_curve + stderr, fliplr(mean_curve - stderr)], ...
         [0.4 0.4 1], 'FaceAlpha', 0.3, 'EdgeColor', 'b');

    % Mean curve
    h_mean = plot(1:N, mean_curve, 'b-', 'LineWidth', 1);

    % Theoretical mean line
    h_theory = yline(theoretical_mean, 'r--', 'LineWidth', 1);

    % Marker for convergence threshold
    if ~isnan(k_star)
        h_thresh = plot(k_star, mean_curve(k_star), 'ko', 'MarkerFaceColor', 'r');
        threshold_str = sprintf('k^* = %d', k_star);
    else
        h_thresh = plot(N, mean_curve(end), 'ko', 'MarkerFaceColor', 'y');
        threshold_str = 'Threshold not reached';
    end

    % Labels and legend
    xlabel('Number of Samples (k)');
    ylabel('Empirical Mean of \phi(x, y)');
    title(sprintf('N = %d (%s)', N, threshold_str), 'FontWeight', 'bold');
    legend([h_trials(1), h_mean, h_theory, h_thresh], ...
        {'Empirical Realizations', 'Mean Across Trials', ...
         'Theoretical Integral', threshold_str}, ...
         'Location', 'northeast', 'FontSize', 7);
    box on; grid on;
end

% Global figure title
sgtitle('\bfConvergence under RWP-Inspired Sampling (\phi(x,y) = x \cdot y)', 'FontSize', 14);

% --- Rejection Sampling Function ---
function [x, y] = rejection_sample(n, pdf, x_range, y_range)
    max_val = 1;
    x = zeros(n, 1); y = zeros(n, 1); count = 0;
    while count < n
        x_try = rand(); y_try = rand(); u = rand();
        if u < pdf(x_try, y_try) / max_val
            count = count + 1;
            x(count) = x_try; y(count) = y_try;
        end
    end
end
