% mode = 1, it shows the execution considering only the projected points
% mode = 2, it shows the execution considering also the projections step
% mode = 3, both mode 1 and mode 2
function [] = plot_execution(Q, q, l, u, a, b, x_s, y_s, mode, interactive)
    f = @(x,y) x^2*Q(1,1) + x*y*(Q(2,1) + Q(1,2)) + y^2*Q(2,2) + q(1)*x + q(2)*y;
    lin = @(x) (b-a(1)*x)/a(2);
    hold on;
    [x_l, x_u] = get_plot_extrems(l, u, lin, x_s, y_s);
    fcontour(f, [x_l x_u]);
    rectangle('Position', [l' (u-l)']);
    fplot(lin, [x_l x_u], "Color", "black");
    n = size(x_s);
    for i = 1:n(2)-1
        if interactive
            pause;
        end 
        p = x_s(:,i:i+1);
        pre = y_s(:,i:i+1);

        if mode == 3 && ~all(pre(:,2) == p(:,2))
            plot(pre(1,:), pre(2,:), "b*");
            line( [p(1,1), pre(1,2)], [p(2,1),pre(2,2)], 'Color', 'blue');
            if interactive
                pause;
            end
            line( [pre(1,2), p(1,2)], [pre(2,2),p(2,2)], 'Color', 'blue');
        end

        if mode == 1 || mode == 3
            plot(p(1,:), p(2,:), "r*");
            line( p(1,:), p(2,:), 'Color','red');
        end

        if mode == 2
            if ~all(pre(:,2) == p(:,2))
                line( [p(1,1), pre(1,2)], [p(2,1),pre(2,2)], 'Color', 'red');
                plot(pre(1,:), pre(2,:), "r*");
                if interactive
                    pause;
                end
                line( [pre(1,2), p(1,2)], [pre(2,2),p(2,2)], 'Color', 'red');
                plot(p(1,:), p(2,:), "r*");
            else
                plot(p(1,:), p(2,:), "r*");
                line( p(1,:), p(2,:), 'Color','red');
            end
        end
    end    
end


function [x_l, x_u] = get_plot_extrems(l, u, lin, x_s, y_s)
    x_l = min(reshape([l, x_s, y_s],1,[]));
    x_u = max(reshape([u, x_s, y_s],1,[]));
    x_l = min(x_l, lin(x_u));
    x_u = max(x_u, lin(x_l));
end