# %% [markdown]
'''
# How to obtain MissingNo. offline in PokéClicker
'''
# %% [markdown]
'''
PokéClicker is a famous online clicker game which attracted a lot of players recently.
Today we will focus on a specific pokemon that can be obtained through a surprising way, MissingNo.
'''
# %% [markdown]
'''
## Who is this pokemon ?

In the pokemon serie, there was a strange monster that appeared to some of the players, MissingNo.
It could not be encountered in the wild, but rather after a bug like the [old man glitch](https://bulbapedia.bulbagarden.net/wiki/Old_man_glitch).
This pokemon resulted from a wrong pokédex entry during a fight, he is actually not unique and has [a lot of variations](https://bulbapedia.bulbagarden.net/wiki/MissingNo.).

<img src="imgs/missingno/sprite.jpg" alt="drawing" width="500"/>

## MissingNo. in PokéClicker

This pokémon was added to PokéClicker mostly as an easter egg and cannot be obtainable through normal manner.
He is used for error handling (if there is a bug within the game), and his pokédex entry is #0.
Hopefully, we can easilly access the game's data to found a way to obtain it.

## Obtaining the pokmon using the javascript console

Pokéclicker is an open-source, community driven game that is mostly developped using Javascript.
The source code is [available on github](https://github.com/pokeclicker/pokeclicker) if you are curious.

Most of the important functions are public, hence is it really easy to access the in-game memory using any web browser.
Indeed, nowadays most if not all browser acts as an IDE and hence have access to a debug environement. Thanks to that, anyone can modify the data completely offline!
For the rest, I will provide the instructions for firefox only, but it should be easilly transferrable to chrome, or edge.

1. First make a save of your game, just in case.
2. Then simply press `F12` to access the web console, or click on the settings icon (up right) then `More tools/Web Developer Tools`.
3. You can now use the public function to gain a pokemon, using `id=0` as a parameter:

```bash
App.game.party.gainPokemonById(0)
```

Congratulations, you just obtained MissingNo. !

>**Note**:
>
>I should not tell you that, but you can of course change the id to another pokémon...
>
>There is also another optionnal parameter. If you want it to be shiny, put the parameter to `1`:
>```bash
>App.game.party.gainPokemonById(0, 1)
>```
'''
# %% [markdown]
'''
## To go further
'''
# %% [markdown]
'''
Obviously this is not the only function that you can have access to!
If you want to cheat, after typing `App.game` you will see the list of available class/functions.
Check for example `App.game.wallet.gainMoney` 💰 😈 💰

Use it sporadically (or not at all) if you don't want to ruin your gameplay.
'''
# %% [markdown]
'''
# Tags
'''
# %% [markdown]
'''
Software-Development; Video-Games
'''