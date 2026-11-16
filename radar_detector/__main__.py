from .cli import main

if __name__ == "__main__":  # guard matters: sweep worker processes re-import the main module
    main()
