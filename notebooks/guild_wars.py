# %% [markdown]
'''
# Optimize jewelry selling in Guild Wars 2™
'''
# %% [markdown]
'''
Guild Wars 2™ (GW2) is a massively multiplayer online role-playing game (MMORPG) developed by ArenaNet and published by NCSoft and ArenaNet.
The game takes place in the world of Tyria, with the re-emergence of a guild dedicated to fighting dragons. 
The Guild Wars series is known to break some classical aspects of MMOs. For example it features a wide solo component (the storyline adapts to the player's actions), or break the Tank/DPS/Healer triptych in player vs player (PvP) and player vs monster (PvM) mode.

In this post, I will talk about the [Guild Wars 2 api](https://wiki.guildwars2.com/wiki/API:Main) developped by ArenaNet for the users, more specifically the python interface by [JuxhinDB](https://github.com/JuxhinDB).
This allows anyone to access the in-game live data like items current selling price, buying prices etc..
Lot of fan-made websites uses the gw2 api to automatize their tools (like https://www.gw2bltc.com, https://www.gw2tp.com or https://www.gw2spidy.com).
'''
# %% [markdown]
'''
## TL;DR
'''
# %% [markdown]
'''
1. The gw2 api can be used to get live trading market data.
2. Having the buying/selling information allows to calculate profitability for any item (given the ressources price).
'''
# %% [markdown]
'''
## Jewelvry in guild wars 2
'''
# %% [markdown]
'''
In my gamer life, my favourite games were always RPGs style games.
That's why I was always attracted, and I was able to play some mmo like [dofus](https://www.dofus.com/) and [Final Fantasy XIV](https://www.finalfantasyxiv.com/) (check this [community api](https://xivapi.com/docs/Welcome)).  
My goal was always to focus on ressources gathering and crafting, I could not therefore miss the job system of GW2 so I became a [jeweler](https://wiki.guildwars2.com/wiki/Jeweler).
I realized quickly that it was not easy to seel my goods and make lot of money out of it.

The process to optimize the selling price was cumbersome because I had to check some website manually like https://www.gw2bltc.com/en/item/19697-Copper-Ore.
So I decided to run some automatic simulations base from grabbed data.
'''
# %% [markdown]
'''
## gw2api usage
'''
# %% [markdown]
'''
<img src="imgs/guild_wars/professions.jpg" alt="drawing" width="800"/>

There is a community based [interface in python](https://github.com/JuxhinDB/gw2-api-interface) by [JuxhinDB](https://github.com/JuxhinDB).
The [installation](https://github.com/JuxhinDB/gw2-api-interface#installation) is quite easy with python-pip.
'''
# %% [code]
### imports
import json
import numpy as np
from gw2api import GuildWars2Client
# %% [markdown]
'''
We can create a new client to the api with,
'''
# %% [code]
# we will create a client like this
gw2_client = GuildWars2Client(version='v2')
# %% [markdown]
'''
This allow us to gather information from an object id.
For example the id for a copper ore can be found in the [official wiki on the right](https://wiki.guildwars2.com/wiki/Copper_Ore).
'''
# %% [code]
# get item by its id
id_object = 19697
gw2_client.items.get(ids={"%5d" %id_object})
# %% [markdown]
'''
A really important feature here is the possibility to get the trading market data for this item,
'''
# %% [code]
# access data from the trading market
gw2_client.commerceprices.get()
# check for a specific item
id_object = 19697
gw2_client.commerceprices.get(ids={"%5d" %id_object})
# %% [markdown]
'''
> **Note**:  
> Be careful to not overload the server by sending [too many requests](https://forum-en.gw2archive.eu/forum/community/api/HEADS-UP-rate-limiting-is-coming).  
> If so, you will get the response "HTTP 429 Limit Exceeded".
'''
# %% [markdown]
'''
## Profitability calculator
'''
# %% [markdown]
'''
Using the trading market data, it is possible to get the optimal object to sell, given the current price of the gathered ressources.
For example a copper earring (master) can be crafted using : x1 Adorned Jewel, x1 Copper Setting (x4 Copper Ores) and x1 Copper Hook (x4 Copper Ores).
We can calculate the resulting price for a [fine Tiger's Eye Copper Stud](https://www.gw2bltc.com/en/item/13268-Tigers-Eye-Copper-Stud) for example,
'''
# %% [code]
# Tiger's Eye price
id_object = 24467
tiger_eye_price = gw2_client.commerceprices.get(ids={"%5d" %id_object})[0]['sells']['unit_price']   
# copper price
id_object = 19697
copper_price = gw2_client.commerceprices.get(ids={"%5d" %id_object})[0]['sells']['unit_price'] 
# craft price
craft_price = tiger_eye_price + 8*copper_price
# %% [markdown]
'''
The rentability of this item is simply the actual price divided by the crafted price,
'''
# %% [code]
### Rentability calculator
id_object = 13289 
print(gw2_client.items.get(ids={"%05d" %id_object})[0]['name'])
actual_price = gw2_client.commerceprices.get(ids={"%5d" %id_object})[0]['sells']['unit_price']
rentability = 100 - 100*craft_price/actual_price
print("Rentability is : %1.2f%%" %rentability)
# %% [markdown]
'''
Finding the id of an item manually is cumbersome. Hopefully, some good guys already created a json file with all the matching id
and name : http://api.gw2tp.com/1/bulk/items-names.json.
'''
# %% [code]
# loading the json file
with open('data/guild_wars/items-names.json') as data_file:    
    data = json.load(data_file)['items']
# %% [markdown]
'''
The following object was designed to calculate the profitability of a given accessory:
'''
# %% [code]
### Profitability for the given accessory
class cAccessory():
    def __init__(self, gem='Amber', ore='Copper', accessory_type='Stud', mastercraft=False):
        self.aName = gem + " " + ore + " " + accessory_type + " " + " m"[mastercraft]
        self.aGem = gem
        self.aOre = ore
        self.aType = accessory_type
        self.aIsMastercrafted = mastercraft
        self.aGemSuffix = None
        if ore == "Copper":
            self.aGemSuffix = "Pebble" 
        elif ore == "Silver":
            self.aGemSuffix = "Nugget" 
        elif ore == "Gold":
            self.aGemSuffix = "Lump" 
        self.aId = None
        self.get_data()
        self.mFindId()
    
    def get_data(self):
        with open('data/guild_wars/items-names.json') as data_file:    
            self.data = json.load(data_file)['items']
    def mFindId(self):
        Id = []
        for i in range(len(self.data)):
            obj_name = self.data[i][1]
            if (obj_name == self.aGem + " " + self.aOre + " " + self.aType):
                Id += [self.data[i][0]]
        if(len(Id) > 0): 
            # We hypothetize that master work have always greater ids
            if(self.aIsMastercrafted and len(Id) > 1):
                self.aId = Id[1]
            else:
                self.aId = Id[0]
    
    def mCraftPrice(self):
        ore_id = None
        gem_id = None
        craft_price = None
        
        for i in range(len(self.data)):
            obj_name = self.data[i][1]
            # Search ore name
            if (obj_name == self.aOre + " Ore"):
                ore_id = self.data[i][0]
            # Search gem name
            elif (obj_name == self.aGem + " " + self.aGemSuffix): 
                gem_id = self.data[i][0]
            # exception just for Pearls..........
            elif (self.aGem == "Pearl"):
                if(obj_name == self.aGem): 
                    gem_id = self.data[i][0]
            
        if(ore_id != None and gem_id != None):
            ore_price = gw2_client.commerceprices.get(ids={"%5d" %ore_id})[0]['sells']['unit_price']
            gem_price = gw2_client.commerceprices.get(ids={"%5d" %gem_id})[0]['sells']['unit_price']
            if (self.aType == 'Stud') | (self.aType == 'Earring'):
                craft_price = gem_price + 8*ore_price
            elif self.aType == 'Ring':
                craft_price = gem_price + 10*ore_price
            elif self.aType == 'Amulet':
                craft_price = gem_price + 12*ore_price
            
            # masterwork are more expensive because of filigrees
            if self.aIsMastercrafted == True:
                craft_price = craft_price + 4*ore_price 
        
        return craft_price
    
    def mSellPrice(self):
        if self.aId is None:
            raise Exception("No id found for {}".format(self.aGem + " " + self.aOre + " " + self.aType))
        sell_price = gw2_client.commerceprices.get(ids={"%5d" %self.aId})[0]['sells']['unit_price']
        
        return sell_price
    
    def mRentability(self):
        if self.aId is None:
            raise Exception("No id found for {}".format(self.aGem + " " + self.aOre + " " + self.aType))
        rent = 100 - 100*self.mCraftPrice() / self.mSellPrice()
        
        return rent
    
    def mGain(self):
        if self.aId is None:
            raise Exception("No id found for {}".format(self.aGem + " " + self.aOre + " " + self.aType))
        gain = self.mSellPrice() - self.mCraftPrice()
        
        return gain
# %% [markdown]
'''
For example to get the current profitability of a [fine Amber Copper Stud](https://www.gw2bltc.com/en/item/13364-Amber-Copper-Stud),
'''  
# %% [code]      
t = cAccessory(gem='Amber', ore='Copper', accessory_type='Stud', mastercraft=False)
print("Rentability for %s is : %1.2f%%" %(gw2_client.items.get(ids={"%5d" %t.aId})[0]['name'], t.mRentability()))
print("Gain for %s is : %3d br" %(gw2_client.items.get(ids={"%5d" %t.aId})[0]['name'], t.mGain()))
# %% [markdown]
'''
The following algorithm can be used to check all jewelry profitability, by specifying from which gems/ores/accessory to choose from:
'''
# %% [code]
### Profitability for gold-based accessories
list_gems = ["Amethyst", "Sunstone", "Topaz", "Carnelian", "Lapis", "Peridot", "Spinel"]   
list_ores = ["Gold"]
list_accessories = ["Earring", "Ring", "Amulet"]
list_mastercraft = [True, False]

res = []
gains = []
names = []
for gem in list_gems:
    for ore in list_ores:
        for accessory in list_accessories:
            for mastercraft in list_mastercraft:
                acc = cAccessory(gem, ore, accessory, mastercraft)
                if acc.aId is not None:
                    names += [acc.aName]
                    res += [acc.mRentability()]
                    gains += [acc.mGain()]

# Sort the gains by decreasing order
print("-------")
print("Accessories:")
for idx in np.argsort(gains)[::-1]:
    print("{}, marginal gain: {}".format(names[idx], gains[idx]))
# %% [markdown]
'''
The [Sunstone Gold Ring](https://www.gw2bltc.com/en/item/45906-Sunstone-Gold-Ring) seems to be the best and more profitable item.
Of course, the marginal gain cannot be the only factor, the frequency of sold item is also important for example.
'''
# %% [markdown]
'''
## Conclusion
''' 
# %% [markdown]
'''
We saw that using the gw2 api allow to get the live market price of any items.
The algorithm I designed allowed me to optimize my selling price, hence gaining lot of money in the game.

I think sharing ther live game data with the community is amazing, a big thumbs up to the GW2 dev team for that!
It is sadly not so frequent in the video game industry, because too many are ultranationalist-centered (hello japanese games!)...
'''
# %% [markdown]
'''
## To go further
''' 
# %% [markdown]
'''
I did not discuss it here but it is of course possible to authenticate to the api.
This to allow the user to access its character content like the inventory, its guild etc...

It can be done with,
```python
client = GuildWars2Client(api_key='API_KEY_VALUE_HERE')
```
'''
# %% [markdown]
'''
## Tags
'''
# %% [markdown]
'''
Software-Development; Video-Games;
'''