dimensionalities = [100, 300, 1000, 5000 10000];

% vector of timing for fixed step size
fixed_timing_100 = [1.084006e-02, 1.222169e-02, 1.385670e-02, 1.623224e-02, 1.498239e-02, 2.454957e-02];
fixed_timing = [mean(fixed_timing_100), 3.590035e-02, 1.186103e+00, 2.996689e+01, 1.084384e+02];

% vector of timing for polyak step size
polyak_timing_100 = [1.049083e-02, 2.188802e-02, 2.545656e-02, 2.026503e-02, 2.878615e-02, 7.883645e-02];
polyak_timing = [mean(polyak_timing_100), 4.896072e-02, 1.186414e+00, 3.232316e+01, 1.056644e+02];

figure('DefaultAxesFontSize',18);
plot(dimensionalities, fixed_timing, 'LineWidth',1);
hold on
plot(dimensionalities, polyak_timing);
xlabel('Dimensionality of the problem', 'LineWidth',1);
ylabel('Time to converge (s)');
lgd = legend('fixed', 'polyak');
lgd.FontSize = 15;

% fplot(@(x) (0.0011)*x.*log(x), [100, 10000]);