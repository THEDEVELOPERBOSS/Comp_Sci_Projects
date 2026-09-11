# Selection UI 

from pick import pick

title = "Please choose your favorite programming language: "
options = ["Python", "JavaScript", "Go", "Rust", "C++"]

# pick() returns a tuple: (selected_option, index)
option, index = pick(options, title, indicator="=>", default_index=0)

print(f"You selected: {option} (Index: {index})")


# How to turn functions on and off 

def dropout(use_dropout):

    if use_dropout:
        return tf.keras.layers.Dropout(
            0.5
        )

    return None