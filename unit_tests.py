from parameters import Parameters, validate_jammer_strategy

def test_create_parameters():

    p_max = 2
    alpha = 1
    sigma_squared = 0.01
    p_recv = 1

    params = Parameters(
        k = 4,
        m = 7,
        p_avg = 0.83 * p_max,
        p_max = p_max,
        c = 50,
        l = 25,
        n = 1, 
        alpha = alpha,
        sigma_squared = sigma_squared,
        p_recv = p_recv,
        rates = [6, 9, 12, 18, 24, 36, 48, 54],
    )

    print(repr(params))
    print(params)

def main():
    test_create_parameters()

if __name__ == "__main__":
    main()