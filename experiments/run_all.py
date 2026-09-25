import e1_taylor
import e2_roots
import e3_linear
import e4_regression


def main():
    for module in (e1_taylor, e2_roots, e3_linear, e4_regression):
        module.main()


if __name__ == "__main__":
    main()
