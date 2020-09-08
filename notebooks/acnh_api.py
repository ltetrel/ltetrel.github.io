# %% [markdown]
'''
# Using python with the Animal Crossing™ New Horizons REST api
'''
# %% [markdown]
'''
Animal Crossing is a well known series exclusive to Nintendo. Originally called "Animal Forest" when it first released on the Nintendo 64 in 2001, it is a simulation game where the player takes the role of a human managing a countryside village of animals.
The newest title Animal Crossing: New Horizons (ACNH) on the Switch system has been a huge success for Nintendo, being [the second most sold title on the Nintendo Switch](https://www.nintendo.co.jp/ir/en/finance/software/index.html) (Covid-19 confinement had a lot to do with it).
There is a huge community behind this title, most notably through the amazing [discord forum](https://discord.com/invite/acnh).

Whenever a title gains popularity, it attracts a lot of talented people developing open-source companion softwares to help others.  
Here I wil talk about the free and open source RESTful [ACNH API](https://github.com/alexislours/ACNHAPI) developped mostly by [alexislours](https://github.com/alexislours).
It is for example used in the really nice [AC helper app](https://github.com/Dimillian/ACHNBrowserUI) I am using.
'''
# %% [markdown]
'''
## tl;dr
1. ACNH api is a free RESTful API for all Animal Crossing New Horizons content.
2. Using the request lib, it is possible to get the content through the ACNH api endpoint.
'''
# %% [markdown]
'''
## RESTful api
'''
# %% [markdown]
'''
Representational state transfer (REST) is a software architectural style initiated by [Roy Fielding](https://roy.gbiv.com/), that defines a set of constraints to be used for creating Web services <cite> fielding2000architectural </cite>.
When using a RESTful api, you will usually make requests to specific URLs, and get the relevant data back in the response.  
This has many advantages: scalability, flexibility, portability and low resources usage (an important criteria when developing mobile apps).
One of its main disadvantage is that it is heavily dependent on the server and the internet connection.

There are four data transactions in any REST system (and HTTP specs): POST (create), GET (read), PUT (edit) and DELETE.
Here, we will mostly use GET requests.
'''
# %% [markdown]
'''
## ACNH API
'''
# %% [markdown]
'''
The [ACNH api](http://acnhapi.com/) is a free RESTful API for critters, fossils, art, music, furniture and villagers from Animal Crossing: New Horizons.
No authentification is needed, and all the data will be available in the [json format](https://www.json.org/json-en.html).

For the following, we will use the [python requests library](https://requests.readthedocs.io/en/master/).
'''
# %% [code]
### imports
import requests
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
# %% [markdown]
'''
We start by creating a function to create POST requests, adn get the result as a dict.
'''
# %% [code]
# Create a GET request on the server, and get the response as a json object
def request_json(https):
    resp = requests.get(https)
    if resp.status_code != 200:
        # This means something went wrong.
        raise Exception('{}: Error {}'.format(https, resp.status_code))

    return resp.json()
# %% [markdown]
'''
Let's list all the fishes available in ACNH, and take one randomly.
'''
# %% [code]
# Get the information from a random fish
# fixing the random state
seed = 0
np.random.seed(seed)
# getting json information from the api
api_endpoint = 'http://acnhapi.com/v1/{}'
fishes = request_json(api_endpoint.format('fish'))
fish_name = np.random.choice(list(fishes.keys()))
seleted_fish = fishes[fish_name]
print(seleted_fish)

# %% [markdown]
'''
It is of course possible to show the graphical content of this fish:
'''
# %% [code]
# read the fish image and icon in a numpy array
fish_img = plt.imread(seleted_fish['image_uri'])
fish_icon = plt.imread(seleted_fish['icon_uri'])
# %% [code]
### Plot the images
# crop image
def crop(rgb_img):
    min_x = np.min(np.nonzero(rgb_img)[1])
    max_x = np.max(np.nonzero(rgb_img)[1])
    min_y = np.min(np.nonzero(rgb_img)[0])
    max_y = np.max(np.nonzero(rgb_img)[0])
    return rgb_img[min_y:max_y, min_x:max_x]
# plot
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(crop(fish_img))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(fish_icon)
plt.xlim(0, 500)
plt.ylim(300, -250)
plt.axis('off')
plt.show()
# %% [markdown]
'''
We can make smart search through the data, for example listing all peppy villagers.
'''
# %% [code]
# getting json information from the api
villagers = request_json(api_endpoint.format('villagers'))
peppy_villagers = []
peppy_imgs = []
for villager in villagers:
    if villagers[villager]['personality'] == 'Peppy':
        peppy_imgs += [villagers[villager]['image_uri']]
        peppy_villagers += [villagers[villager]['name']['name-EUfr']]
print('Here is the list of all peppy villagers : {}'.format(peppy_villagers))
# %% [markdown]
'''
And their posters,
'''
# %% [code]
### All posters of peppy villager
n_rows = int(np.ceil(np.sqrt(len(peppy_imgs))))
n_cols = int(n_rows)
size_img = plt.imread(peppy_imgs[0]).shape
grid_peppy = np.zeros((n_rows*size_img[0], n_cols*size_img[1], size_img[-1] - 1))
for i in range(n_rows):
    for j in range(n_cols):
        idx = [i*size_img[0], (i+1)*size_img[0], j*size_img[1], (j+1)*size_img[1]]
        grid_peppy[idx[0]:idx[1], idx[2]:idx[3], :3] = plt.imread(peppy_imgs[i*n_rows + j])[:, :, :3]
# plot
plt.imshow(grid_peppy)
plt.axis('off')
plt.show()
# %% [markdown]
'''
Finally, let's analyze the dominant colors in all ACNH houseware items.
'''
# %% [code]
### Color frequencies for all houseware items
# get the item's color
items = request_json(api_endpoint.format('houseware'))
colors = np.array([])
for item in items:
    for remake in items[item]:
        colors = np.append(colors, remake['color-1'])
        colors = np.append(colors, remake['color-2'])
#create legend
cmap = np.unique(colors)
cmap_plt = np.copy(cmap)
for i in range(len(cmap_plt)):
    if (cmap_plt[i] == 'Colorful') | (cmap_plt[i] == 'None') :
            legend = 'cyan'
    else:
        legend = cmap_plt[i].lower().replace(' ', '')
    cmap_plt[i] = legend
#plot
fig = go.Figure()
for i in range(len(cmap)):
    fig.add_trace(go.Histogram(histfunc="count", x=colors[colors == cmap[i]], marker_color=cmap_plt[i], name=cmap[i]))
fig.update_layout(title='ACNH colors for houseware items')
fig.show(renderer="iframe_connected", config={'showLink': False})
# %% [markdown]
'''
The most used color for houseware items in AC NH is the white.
'''
# %% [markdown]
'''
## Tags
'''
# %% [markdown]
'''
Software-Development; Video-Games;
'''