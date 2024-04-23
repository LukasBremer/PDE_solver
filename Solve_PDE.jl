using DynamicalSystems, OrdinaryDiffEq, CairoMakie,Random, LinearAlgebra

function Fitzhugh_Nagumo_reaction(x,p,t)
    a, b, epsilon  = p
    u,w = x
    du =@. a * u * (u - b) * (1 - u) - w  
    dw =@. epsilon * (u - w)
    return SVector(du,dw)*0
end

function Euler_step(u,w,fun,p)
    return @.[u,w] + delta_t * fun([u,w],p,0)
end

function ODE_step(x,fun,p)
    diffeq = (alg = Tsit5(), adaptive = false, delta_t=0.003)
    ds = @. ContinuousDynamicalSystem(fun,x,p ;diffeq)
    step!(ds,1)
    x = current_state(ds)
    t = current_time(ds)
    return x,t 
end 

function ghost_layer(x_grid)
    extended_matrix = zeros(size(x_grid))
    original_matrix = x_grid[2:end-1, 2:end-1]
    extended_matrix[2:end-1, 2:end-1] = original_matrix
    extended_matrix[2:end-1, 2:end-1] = original_matrix
    extended_matrix[1, 2:end-1] .= original_matrix[1, :]
    extended_matrix[end, 2:end-1] .= original_matrix[end, :]
    extended_matrix[:, 1] .= extended_matrix[:, 2]
    extended_matrix[:, end] .= extended_matrix[:, end-1]
    return extended_matrix
end

function initialize_gaussian(n)

    # Erstelle eine leere 100x100 Matrix mit Nullen
    matrix = zeros(Float64, n, n)

    # Parameter für die Gaussfunktionen
    μ1 = [30, 30]  # Mittelpunkt der ersten Gaussfunktion
    σ1 = [3, 3]  # Standardabweichung der ersten Gaussfunktion
    μ2 = [100, 100]  # Mittelpunkt der zweiten Gaussfunktion
    σ2 = [3, 3]  # Standardabweichung der zweiten Gaussfunktion

    # Funktion zur Berechnung der Gaussfunktion
    gaussian(x, μ, σ) = exp(-sum((x .- μ).^2 ./ (2 * σ.^2)))

    # Iteriere über die Matrix und wende die Gaussfunktionen an
    for i in 1:n
        for j in 1:n
            # Berechne den Wert der ersten Gaussfunktion an der Position (i, j)
            value1 = gaussian([i, j], μ1, σ1)
            # Berechne den Wert der zweiten Gaussfunktion an der Position (i, j)
            value2 = gaussian([i, j], μ2, σ2)
            # Addiere die Werte der Gaussfunktionen zur Matrix
            matrix[i, j] = value1 + value2
        end
    end
    return matrix
end

function initialize_u(n)

    # Erstelle eine leere 300x300 Matrix mit Nullen
    matrix = zeros(Float64, n, n)

    # Setze die untere Hälfte der Matrix auf Einsen
    matrix[:,Integer(n/4):end] .= 1
    return Float64.(matrix)
end

function initialize_w(n)
    # Erstelle eine leere 300x300 Matrix mit Nullen
    matrix = zeros(Float64, n, n)

    # Bestimme die Größe des unteren linken Viertels
    lower_left_size = div(n, 2)

    # Setze das untere linke Viertel auf 0.5
    #matrix[end-lower_left_size+1:end, 1:lower_left_size] .= .2
    matrix.=.2
    # Zeige die erstellte Matrix
    return Float64.(matrix)
end

function laplace_9ps(x_y_grid)
    matrix = x_y_grid
    nine_point_stencil = [1 4 1; 4 -20 4; 1 4 1]
    laplace = zeros(size(matrix))

    for i in -1:1
        for j in -1:1
            alpha = nine_point_stencil[2-i,2-j]
            laplace = laplace + alpha * circshift(matrix,(i,j))         
        end
    end
    return laplace/(6*h^2)
end

function step_PDE(u,w,fun)
    
    u,w = Euler_step(u,w, fun ,par)
    u = u + delta_t * d * laplace_9ps(u)
    # du,dw = fun([u,w],par,0)
    # du += d*laplace_9ps(u)

    # u += du
    # w += dw
    return u,w
end

function PDE_trajectory(u,w,iterations,fun)
    u = [u]
    w = [w]

    for i in 1:iterations 
        u_temp, w_temp = step_PDE(u[end],w[end],fun)
        u_temp ,w_temp = ghost_layer(u_temp), ghost_layer(w_temp)
        if i%100==0
            u = push!(u,u_temp)
            w = push!(w,w_temp)
        else
        end
    end

    return u,w
end

delta_t = .003
d = 1
par = [3,.2,.01]
N = 150
h = 2


u_0,w_0 = initialize_gaussian(N), initialize_w(N)
u_list,w_list = PDE_trajectory(u_0,w_0,100000,Fitzhugh_Nagumo_reaction)
length(u_list)
heatmap(u_list[end])



heatmap(u_0 + delta_t * laplace_9ps(u_0))

