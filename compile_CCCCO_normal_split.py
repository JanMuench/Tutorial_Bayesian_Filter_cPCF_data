
import pickle
import pystan




def save(obj, filename, regress_code_kinetic):
    """Save compiled models for reuse."""


    with open(filename+".pic", 'wb') as pointer:
        pickle.dump(obj, pointer, protocol=pickle.HIGHEST_PROTOCOL)
    with open(filename+".txt", "w") as file:
        file.write(regress_code_kinetic)




def load(filename):
    """Reload compiled models for reuse."""
    import pickle
    return pickle.load(open(filename, 'rb'))

def main():

    file_name_mod = "Kalman_fluorescence.txt"

    with open(file_name_mod, "r") as file:
        model = file.read()

    print("constructiong the stan code")
    regress_code = model
    print(regress_code)

    filename = "Kalman_fluorescence"
    #filename ="moffat_steady_state"
    print(filename + ".txt")
    print("Compile now \n" + regress_code)
    extra_compile_args =["-pthread", "-DSTAN_THREADS"]
    model = pystan.StanModel(model_code =regress_code,
                             extra_compile_args=extra_compile_args)

    save(model, filename, regress_code)
    print("Finished")

if __name__ == "__main__":
    main()
