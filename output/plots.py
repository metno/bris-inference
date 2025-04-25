import datetime
from math import sqrt
from anemoi.training.diagnostics.maps import Coastlines
from anemoi.training.diagnostics.plots import EquirectangularProjection
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from typing import Optional
import scipy
import torch
import xarray as xr
import os
from torch import nn
from matplotlib import colormaps
import random
from pyshtools.expand import SHExpandGLQ, SHGLQ
import numpy as np
import pylab as pyl




# from aifs.model.losses import limitedareaMSEloss, WeightedMSELoss

def setaxsettings(fig, ax, title, ds1, ds1_era, pl, g):
    lat = np.append(ds1["latitude"].to_numpy(), ds1_era["longitude"].to_numpy())
    lon = np.append(ds1["longitude"].to_numpy(), ds1_era["latitude"].to_numpy())
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat) 
    ax.set_xlim((-0.2, 0.5))
    ax.set_ylim((0.7, 1.05))
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='black')
    ax.text(0.35, 0.73, str(ds1.isel(time=0)["time"].to_numpy())[:19], **text_kwargs)

    vlt = str(((ds1.isel(time=0)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    ax.text(0.35, 0.71,f'lead time: {vlt} h', **text_kwargs)
    if pl is not None:
        uwind = ds1.isel(time=0, pressure=pl)["x_wind_pl"].to_numpy()
        vwind = ds1.isel(time=0, pressure=pl)["y_wind_pl"].to_numpy()
        uwind_era = ds1_era.isel(time=0, pressure=pl)["x_wind_pl"].to_numpy()
        vwind_era = ds1_era.isel(time=0, pressure=pl)["y_wind_pl"].to_numpy()
        windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
        windspeed_era = np.sqrt(np.square(uwind_era) + np.square(vwind_era))
        t = np.append(windspeed, windspeed_era)
    else:
        uwind = ds1.isel(time=0)["x_wind_10m"].to_numpy()
        vwind = ds1.isel(time=0)["y_wind_10m"].to_numpy()
        uwind_era = ds1_era.isel(time=0)["x_wind_10m"].to_numpy()
        vwind_era = ds1_era.isel(time=0)["y_wind_10m"].to_numpy()
        windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
        windspeed_era = np.sqrt(np.square(uwind_era) + np.square(vwind_era))
        t = np.append(windspeed, windspeed_era)
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe=g)
    ax.set_title("\n".join(wrap(title, 60)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    fig.colorbar(psc, ax=ax)

def animationUpdate(ds1, ds1_era, fig, ax, num, pl, g):
    for txt in ax.texts:
        txt.set_visible(False)
    lat = np.append(ds1["latitude"].to_numpy(), ds1_era["longitude"].to_numpy())
    lon = np.append(ds1["longitude"].to_numpy(), ds1_era["latitude"].to_numpy())
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat) 
    if pl is not None:
        uwind = ds1.isel(time=num, pressure=pl)["x_wind_pl"].to_numpy()
        vwind = ds1.isel(time=num, pressure=pl)["y_wind_pl"].to_numpy()
        uwind_era = ds1_era.isel(time=num, pressure=pl)["x_wind_pl"].to_numpy()
        vwind_era = ds1_era.isel(time=num, pressure=pl)["y_wind_pl"].to_numpy()
        windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
        windspeed_era = np.sqrt(np.square(uwind_era) + np.square(vwind_era))
        t = np.append(windspeed, windspeed_era)
    else:
        uwind = ds1.isel(time=num)["x_wind_10m"].to_numpy()
        vwind = ds1.isel(time=num)["y_wind_10m"].to_numpy()
        uwind_era = ds1_era.isel(time=num)["x_wind_10m"].to_numpy()
        vwind_era = ds1_era.isel(time=num)["y_wind_10m"].to_numpy()
        windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
        windspeed_era = np.sqrt(np.square(uwind_era) + np.square(vwind_era))
        t = np.append(windspeed, windspeed_era)
    scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe=g)
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='black')
    # ax.text(0.35, 0.73, str(ds1.isel(time=num)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=num)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    ax.text(0.35, 0.7,f'lead time: {vlt} h', **text_kwargs)

def setaxsettings_variable(fig, ax, title, ds1, ds1_era, var_name, pl, g): 
    lat = np.append(ds1["latitude"].to_numpy(), ds1_era["longitude"].to_numpy())
    lon = np.append(ds1["longitude"].to_numpy(), ds1_era["latitude"].to_numpy())
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat) 
    ax.set_xlim((-0.2, 0.5))
    ax.set_ylim((0.7, 1.05))
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='black')
    # ax.text(0.35, 0.73, str(ds1.isel(time=0)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=0)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    ax.text(0.35, 0.7,f'lead time: {vlt} h', **text_kwargs)
     # calculate variable and plot figure
    if pl is not None:
        t = np.append(ds1.isel(time=0, pressure=pl)[var_name].to_numpy(), ds1_era.isel(time=0, pressure=pl)[var_name].to_numpy())
    else:
        t = np.append(ds1.isel(time=0)[var_name].to_numpy(), ds1_era.isel(time=0)[var_name].to_numpy())
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe=g)
    ax.set_title("\n".join(wrap(title, 60)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    fig.colorbar(psc, ax=ax)

def animationUpdate_variable(ds1, ds1_era, fig, ax, num, var_name, pl, g):
    for txt in ax.texts:
        txt.set_visible(False)
    lat = np.append(ds1["latitude"].to_numpy(), ds1_era["longitude"].to_numpy())
    lon = np.append(ds1["longitude"].to_numpy(), ds1_era["latitude"].to_numpy())
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat) 
    if pl is not None:
        t = np.append(ds1.isel(time=num, pressure=pl)[var_name].to_numpy(), ds1_era.isel(time=num, pressure=pl)[var_name].to_numpy())
    else:
        t = np.append(ds1.isel(time=num)[var_name].to_numpy(), ds1_era.isel(time=num)[var_name].to_numpy())
    scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe = g)
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='black')
    # ax.text(0.35, 0.68, str(ds1.isel(time=num)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=num)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    ax.text(0.35, 0.7,f'lead time: {vlt} h', **text_kwargs)

def setaxsettings_pl(fig, ax, title, ds1, var_name, pl, g): 
    # calculate lon lat
    lonlat = np.append(np.expand_dims(ds1["latitude"].to_numpy(), axis=1),np.expand_dims(ds1["longitude"].to_numpy(), axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat) 

    # set axis limits
    ax.set_xlim((-0.2, 0.5))
    ax.set_ylim((0.7, 1.05))

    # add date time label 
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='black')
    # ax.text(0.35, 0.73, str(ds1.isel(time=0)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=0)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    ax.text(0.35, 0.7,f'lead time: {vlt} h', **text_kwargs)

    # get scatter plot
    var = ds1.sel(pressure = pl).isel(time=0)[var_name].to_numpy()
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, var, s=10, globe = g)

    # set title, x and y ticklabels and continents
    ax.set_title("\n".join(wrap(title, 60)))
    plt.setp(ax.get_xticklabels(), visible=False)                                                           
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    fig.colorbar(psc, ax=ax)

def animationUpdate_pl(ds1, fig, ax, num, var_name, pl, g):
    for txt in ax.texts:
        txt.set_visible(False)
    lonlat = np.append(np.expand_dims(ds1["latitude"].to_numpy(), axis=1),np.expand_dims(ds1["longitude"].to_numpy(), axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat) 

    # get scatter plot
    var = ds1.sel(pressure = pl).isel(time=num)[var_name].to_numpy()
    scatter_plot(fig, ax, pc_lon, pc_lat, var, s=20, globe = g)
    text_kwargs = dict(ha='center', va='center', fontsize=15, color='black')
    ax.text(0.35, 0.68, str(ds1.isel(time=num)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=num)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    ax.text(0.35, 0.70,f'lead time: {vlt} h', **text_kwargs)

def scatter_plot(fig, ax, lon: np.array, lat: np.array, data: np.array, cmap: str = "RdYlGn", s: Optional[float] = 0.5, globe: Optional[bool] = False, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    """Lat-lon scatter plot: can work with arbitrary grids.

    Parameters
    ----------
    fig : _type_
        Figure object handle
    ax : _type_
        Axis object handle
    lon : np.ndarray
        longitude coordinates array, shape (lon,)
    lat : np.ndarray
        latitude coordinates array, shape (lat,)
    data : _type_
        Data to plot
    cmap : str, optional
        Colormap string from matplotlib, by default "viridis"
    title : _type_, optional
        Title for plot, by default None
    """
    # print(data.min())
    # print(data.max())
    # data = np.where(data<200, np.nan, data)
    if vmin == None:
        minimum = data.min()
        maximum = data.max()
        if globe == False:
            minimum = data[:622521].min()
            maximum = data[:622521].max()
            ax.set_xlim((-0.2, 0.8))
            ax.set_ylim((0.9, 1.3))
    else:
        minimum = vmin
        maximum = vmax
        if globe == False:
            ax.set_xlim((-0.2, 0.8))
            ax.set_ylim((0.9, 1.3))
        

        
    psc = ax.scatter(
        lon,
        lat,
        c=data,
        cmap=cmap,
        # s=[8 if i<4134 else 600 for i in range(len(data))],
        s = s,
        alpha=1.0,
        # marker='s',
        # norm=TwoSlopeNorm(vcenter=0.0) if cmap == "bwr" else None,
        vmin = minimum,
        vmax = maximum, 
        # vmin = -7,
        # vmax = 7,
        rasterized=True,
    )

    return psc

def getWindspeedLossRollout(ds1_truth, ds1, rollouts, num, pl):
    if pl == None:
        # calculate wind speed ground truth
        uwind_t = ds1_truth.isel(time=num)["x_wind_10m"].to_numpy()
        vwind_t = ds1_truth.isel(time=num)["y_wind_10m"].to_numpy()
        windspeed_t = np.sqrt(np.square(uwind_t) + np.square(vwind_t))
        target = torch.Tensor(np.expand_dims(windspeed_t,2))

        # calculate predicted wind speed
        uwind = ds1.isel(time=num)["x_wind_10m"].to_numpy()
        vwind = ds1.isel(time=num)["y_wind_10m"].to_numpy()
        windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
        pred = torch.Tensor(np.expand_dims(windspeed,2))
    else:
        # calculate wind speed ground truth
        # print(ds1_truth.data_vars)
        uwind_t = ds1_truth.sel(pressure = pl).isel(time=num)["x_wind_pl"].to_numpy()
        vwind_t = ds1_truth.sel(pressure = pl).isel(time=num)["y_wind_pl"].to_numpy()
        windspeed_t = np.sqrt(np.square(uwind_t) + np.square(vwind_t))
        target = torch.Tensor(np.expand_dims(np.expand_dims(windspeed_t,0), 2))
        # calculate predicted wind speed
        uwind = ds1.sel(pressure = pl).isel(time=num)["x_wind_pl"].to_numpy()
        vwind = ds1.sel(pressure = pl).isel(time=num)["y_wind_pl"].to_numpy()
        windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
        pred = torch.Tensor(np.expand_dims(np.expand_dims(windspeed,0), 2))
    # Append loss to rollouts dictionary
    if num in rollouts.keys():
        rollouts[num].append(sqrt(RMSE(pred, target).item()))
    else:
        rollouts[num] = [sqrt(RMSE(pred, target).item())]
    return rollouts

def getAverageRollout(folder_path):
    # Calculates the LA wind speed rollout loss over all files in the folder
    rollouts = {}
    for name in os.listdir(folder_path):
        # LIMITED AREA FILES ONLY
        if name[:18]=="w10m_t2m_10km_pred":
            # get the data
            ds1 = xr.open_dataset(os.path.join(folder_path, name))

            # get the ground truth
            name = name[:14] + "truth" + name[18:]
            ds1_truth = xr.open_dataset(os.path.join(folder_path, name))

            # initiate the UNWEIGHTED loss function and append the rollouts for each time step
            loss = limitedareaRMSEloss(area_weights=torch.ones(ds1_truth.isel(time=0)["x_wind_10m"].shape[1]), data_split_index = torch.IntTensor([4134]))
            for num, t in enumerate(ds1["time"]):
                rollouts = getWindspeedLossRollout(ds1, ds1_truth, rollouts, loss, num)
    
    # sort the rollout by date, and take the average over all values
    rolloutitems = sorted(rollouts.items(), key = lambda p: p)
    rolloutavg = []
    for (key, values) in rolloutitems:
        rolloutavg.append(np.sum(values)/len(values))
    return rolloutavg

def getSingleLossRollout(ds1_truth, ds1, rollouts, var_name, pl, num):
    # Calculates the rollout loss of a specific file and rollout step
    if var_name == "windspeed":
        return getWindspeedLossRollout(ds1_truth, ds1, rollouts, num, pl)
    if pl is None:
        var_t = ds1_truth.isel(time=num)[var_name].to_numpy()
        target = torch.Tensor(np.expand_dims(var_t,2))
        var_p = ds1.isel(time=num)[var_name].to_numpy()
        pred = torch.Tensor(np.expand_dims(var_p,2))
    else: 
        var_t = ds1_truth.sel(pressure = pl).isel(time=num)[var_name].to_numpy()
        target = torch.Tensor(np.expand_dims(np.expand_dims(var_t,0),2))
        var_p = ds1.sel(pressure = pl).isel(time=num)[var_name].to_numpy()
        pred = torch.Tensor(np.expand_dims(np.expand_dims(var_p,0),2))
    if num in rollouts.keys():
        rollouts[num].append(sqrt(RMSE(pred, target).item()))
    else:
        rollouts[num] = [sqrt(RMSE(pred, target).item())]
    return rollouts

def PlotVarRolloutLAM(folder_paths, var_name, pl):
    rolloutavgs = []
    rolloutstds = []
    for path in folder_paths:
        rollouts = {}
        for name in os.listdir(path):
            # print(name)
            if name.startswith("w10m_t2m_10km_pred"):
                ds1 = xr.open_dataset(os.path.join(path, name))
                name = name[:14] + "truth" + name[18:]
                ds1_truth = xr.open_dataset(os.path.join(path, name))
                for num, t in enumerate(ds1["time"]):
                    rollouts = getSingleLossRollout(ds1_truth, ds1, rollouts, var_name, pl, num)
        rolloutitems = sorted(rollouts.items(), key = lambda p: p)

        rolloutavg_4 = []
        rolloutstd_4 = []
        for (key, values) in rolloutitems:
            rolloutavg_4.append(np.sum(values)/len(values))
            rolloutstd_4.append(1.96*np.std(values)/np.sqrt(len(values)))
        # print(rolloutstd_4)
        rolloutavgs.append(rolloutavg_4)
        rolloutstds.append(rolloutstd_4)
    leadtimes = []
    for i in range(len(rolloutavgs[0])):
        vlt = str(((ds1.isel(time=i)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
        leadtimes.append(vlt[:-2])
    fig, ax = plt.subplots()

    # ci1 = 1.96 * np.std(rolloutavgs[0][1:])/np.sqrt(len(leadtimes[1:]))
    colours = ["darkred", "red", "orange"]
    for i in range(len(rolloutavgs)):
        ax.plot(leadtimes[1:], rolloutavgs[i][1:], label = list(folder_paths.values())[i], c = colours[i])
        ax.fill_between(leadtimes[1:], (np.array(rolloutavgs[i][1:]) - np.array(rolloutstds[i][1:])), (np.array(rolloutavgs[i][1:])+np.array(rolloutstds[i][1:])), color=colours[i], alpha=.1)
    # ax.plot(leadtimes[1:], rolloutavgs[0][1:], label = list(folder_paths.values())[0], c = "darkred")
    # ax.plot(leadtimes[1:], rolloutavgs[1][1:], label = list(folder_paths.values())[1], c = "red")
    # ax.plot(leadtimes[1:], rolloutavgs[2][1:], label = list(folder_paths.values())[2], c = "darkorange")
    # ax.fill_between(leadtimes[1:], (np.array(rolloutavgs[0][1:]) - np.array(rolloutstds[0][1:])), (np.array(rolloutavgs[0][1:])+np.array(rolloutstds[0][1:])), color='darkred', alpha=.1)
    # ax.fill_between(leadtimes[1:], (np.array(rolloutavgs[1][1:]) - np.array(rolloutstds[1][1:])), (np.array(rolloutavgs[1][1:])+np.array(rolloutstds[1][1:])), color='red', alpha=.1)
    # ax.fill_between(leadtimes[1:], (np.array(rolloutavgs[2][1:]) - np.array(rolloutstds[2][1:])), (np.array(rolloutavgs[2][1:])+np.array(rolloutstds[2][1:])), color='darkorange', alpha=.1)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("lead time (h)")
    start, end = ax.get_xlim()
    plt.xticks(np.arange(start, end, 4))
    ax.legend()
    if pl is None:
        # ax.set_title("\n".join(wrap(f"{var_name} average RMSE over the limited area", 60)))
        ax.set_title("\n".join(wrap(f"RMSE of {var_name} over the limited area", 60)))
    else:
        # ax.set_title("\n".join(wrap(f"{var_name} at {pl} hPa average RMSE over the limited area", 60)))
        # ax.set_title("\n".join(wrap(f"{var_name} at {pl} hPa over the limited area", 60)))
        ax.set_title("\n".join(wrap(f"RMSE of {var_name} over the limited area", 60)))
    plt.savefig(f"{var_name}_{pl}_lam_rollout_avg_plot.png")
    print(f"saved {var_name}_{pl}_lam_rollout_avg_plot.png")



def PlotVarRolloutERA(folder_paths, var_name, pl):
    rolloutavgs = []
    rolloutstds = []
    for path in folder_paths:
        rollouts = {}
        for name in os.listdir(path):
            if name[:19]=="w10m_t2m_1deg_truth" and name!="w10m_t2m_1deg_truth_start-2022-01-08T12_240hforecast.nc":
                ds1_truth = xr.open_dataset(os.path.join(path, name))
                name = name[:14] + "pred" + name[19:]
                if os.path.exists(os.path.join(path, name)):
                    ds1 = xr.open_dataset(os.path.join(path, name))
                    # loss = WeightedRMSELoss(area_weights=torch.ones(ds1_truth.isel(time=0)["x_wind_10m"].shape[1]))
                    for num, t in enumerate(ds1["time"]):
                        rollouts = getSingleLossRollout(ds1_truth, ds1, rollouts, var_name, pl, num)
        rolloutitems = sorted(rollouts.items(), key = lambda p: p)
        rolloutavg_4 = []
        rolloutstd_4 = []
        for (key, values) in rolloutitems:
            rolloutavg_4.append(np.sum(values)/len(values))
            rolloutstd_4.append(1.96*np.std(values)/np.sqrt(len(values)))
        # print(rolloutstd_4)
        rolloutavgs.append(rolloutavg_4)
        rolloutstds.append(rolloutstd_4)
    leadtimes = []
    for i in range(41):
        vlt = str(((ds1.isel(time=i)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
        leadtimes.append(vlt[:-2])
    fig, ax = plt.subplots()
    colours = ["darkred", "red", "orange"]
    for i in range(len(rolloutavgs)):
        ax.plot(leadtimes[1:], rolloutavgs[i][1:], label = list(folder_paths.values())[i], c = colours[i])
        ax.fill_between(leadtimes[1:], (np.array(rolloutavgs[i][1:]) - np.array(rolloutstds[i][1:])), (np.array(rolloutavgs[i][1:])+np.array(rolloutstds[i][1:])), color=colours[i], alpha=.1)
    # ax.plot(leadtimes[1:], rolloutavgs[1][1:], label = list(folder_paths.values())[1], c = "red")
    # ax.plot(leadtimes[1:], rolloutavgs[2][1:], label = list(folder_paths.values())[2], c = "darkorange")
    
    # ax.fill_between(leadtimes[1:], (np.array(rolloutavgs[1][1:]) - np.array(rolloutstds[1][1:])), (np.array(rolloutavgs[1][1:])+np.array(rolloutstds[1][1:])), color='red', alpha=.1)
    # ax.fill_between(leadtimes[1:], (np.array(rolloutavgs[2][1:]) - np.array(rolloutstds[2][1:])), (np.array(rolloutavgs[2][1:])+np.array(rolloutstds[2][1:])), color='darkorange', alpha=.1)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("lead time (h)")
    start, end = ax.get_xlim()
    plt.xticks(np.arange(start, end, 4))
    ax.legend()
    if pl is None:
        ax.set_title("\n".join(wrap(f"{var_name} average global RMSE", 60)))
    else:
        # ax.set_title("\n".join(wrap(f"{var_name} at {pl} hPa average global RMSE", 60)))
        ax.set_title("\n".join(wrap(f"{var_name} average global RMSE", 60)))
    plt.savefig(f"{var_name}_{pl}_era_rollout_avg_plot.png")

def RMSE(pred: torch.Tensor, target: torch.Tensor):
    # print(pred.shape)
    out = torch.square(pred-target)
    return torch.sum(out, dim=(0,1,2))/(out.shape[1])

class limitedareaRMSEloss(nn.Module):
    """Latitude-weighted MSE loss, calculated only over the limited area"""

    def __init__(self, area_weights: torch.Tensor, data_split_index: torch.Tensor, data_variances: Optional[torch.Tensor] = None) -> None:
        """(inverse-)variance-weighted MSE Loss.

        Parameters
        ----------
        area_weights : torch.Tensor
            Weights by area
        data_variances : Optional[torch.Tensor], optional
            precomputed, per-variable stepwise variance estimate, by default None
        """
        super().__init__()
        # self.register_buffer("weights", area_weights, persistent=True)
        # self.register_buffer("dsi", data_split_index, persistent=True)
        self.weights = area_weights
        self.dsi = data_split_index
        if data_variances is not None:
            self.register_buffer("ivar", data_variances, persistent=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, squash=True) -> torch.Tensor:
        """Calculates the lat-weighted MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True

        Returns
        -------
        torch.Tensor
            Limited area latitude weighted MSE loss
        """
        self.weights = self.weights.to("cpu")
        self.dsi = self.dsi.to("cpu")
        pred = pred[ :, :self.dsi, :]
        target = target[:, :self.dsi, :]
        if hasattr(self, "ivar"):
            if squash:
                out = (torch.square(pred - target) * self.ivar).mean(dim=-1)
            else:
                out = torch.square(pred - target) * self.ivar
        else:
            if squash:
                out = torch.square(pred - target).mean(dim=-1)
            else:
                out = torch.square(pred - target)

        if squash:
            out = out * self.weights[:self.dsi].expand_as(out)
            out /= torch.sum(self.weights[:self.dsi].expand_as(out))
            return out.sum()

        out = out * self.weights[..., None].expand_as(out)
        out /= torch.sum(self.weights[..., None].expand_as(out))
        return torch.sqrt(out.sum(axis=(0, 1)))

def SinglePlotWithBoundaries(fig, ax, ds1, ds1_era, var, pl, rolloutstep, title, g, vmin: Optional[float] = None, vmax: Optional[float] = None):
    if var == "windspeed":
        return WindSpeedSinglePlotWithBoundaries(fig, ax, ds1, ds1_era, pl, rolloutstep, title, g)
    lat = np.append(ds1["latitude"].to_numpy(), ds1_era["longitude"].to_numpy())
    lon = np.append(ds1["longitude"].to_numpy(), ds1_era["latitude"].to_numpy())
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)

    # Add Time Stamp
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='black')
    # ax.text(0.35, 0.68, str(ds1.isel(time=0)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=rolloutstep)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    # if g == False:
    #     ax.text(0.35, 0.7,f'lead time: {vlt}h', **text_kwargs)
    # else:
    #     ax.text(2.2, -1.9,f'lead time: {vlt}h', **text_kwargs)

    # calculate variable and plot figure
    if pl is not None:
        t = np.append(ds1.isel(time=rolloutstep, pressure=pl)[var].to_numpy(), ds1_era.isel(time=rolloutstep, pressure=pl)[var].to_numpy())
    else:
        t = np.append(ds1.isel(time=rolloutstep)[var].to_numpy(), ds1_era.isel(time=rolloutstep)[var].to_numpy())
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe=g, vmin=vmin, vmax=vmax)

    ax.set_title("\n".join(wrap(title, 40)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    # fig.colorbar(psc, ax=ax)
    return psc

def SinglePlotNoBoundaries(fig, ax, ds1, var, pl, rolloutstep, title, g, vmin: Optional[float] = None, vmax: Optional[float] = None):
    if var == "windspeed":
        return WindspeedPlotNoBoundaries(fig, ax, ds1, pl, rolloutstep, title, g)
    lat = ds1["latitude"].values.flatten()
    lon = ds1["longitude"].values.flatten()
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)

    # Add Time Stamp
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='black')
    # ax.text(0.35, 0.68, str(ds1.isel(time=0)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=rolloutstep)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    # ax.text(6, ,f'lead time: {vlt}h', **text_kwargs)

    # calculate variable and plot figure
    if pl is not None:
        t = ds1.isel(time=rolloutstep, pressure=pl)[var].values.flatten()
    else:
        t = ds1.isel(time=rolloutstep)[var].values.flatten()
    print(t.shape)
    print(pc_lon.shape)
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe=g, vmin=vmin, vmax=vmax)

    ax.set_title("\n".join(wrap(title, 40)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    fig.colorbar(psc, ax=ax)
    return psc

def SinglePlotZarr(fig, ax, ds1, var, pl, rolloutstep, title, g, vmin: Optional[float] = None, vmax: Optional[float] = None):
    if var == "windspeed":
        return WindspeedPlotNoBoundaries(fig, ax, ds1, pl, rolloutstep, title, g)
    lat = ds1.latitudes
    lon = ds1.longitudes
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)

    # Add Time Stamp
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='black')

    # calculate variable and plot figure
    index = ds1.name_to_index[var]
    t = ds1[rolloutstep, index, :, :].flatten()

    print(t.shape)
    print(pc_lon.shape)
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe=g, vmin=vmin, vmax=vmax)

    ax.set_title("\n".join(wrap(title, 40)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    fig.colorbar(psc, ax=ax)
    return psc
    # plt.savefig(f"{var}_{pl}_single_plot.png")
# def DoublePlotWithBoundaries():
def HAplotwindspeed(fig, ax, ds1, globe):
    uwind = ds1["u"]
    vwind = ds1["v"]
    windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
    psc = scatter_plot(fig, ax, np.radians((ds1["longitude"]+180)%360 -180), np.radians(ds1["latitude"]), windspeed, s=10, globe=globe)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    # fig.colorbar(psc, ax=ax)
    return psc

def HAplot(fig, ax, ds1, var, globe):
    if var == "windspeed":
        return HAplotwindspeed(fig, ax, ds1, globe)
    psc = scatter_plot(fig, ax, np.radians((ds1["longitude"]+180)%360 -180), np.radians(ds1["latitude"]), ds1[var], s=10, globe=globe)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    return psc

def WindSpeedSinglePlotWithBoundaries(fig, ax, ds1, ds1_era, pl, rolloutstep, title, g):
    lat = np.append(ds1["latitude"].to_numpy(), ds1_era["longitude"].to_numpy())
    lon = np.append(ds1["longitude"].to_numpy(), ds1_era["latitude"].to_numpy())
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)

    # Add Time Stamp
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='black')
    # ax.text(0.35, 0.68, str(ds1.isel(time=0)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=rolloutstep)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    # if g == False:
    #     ax.text(0.35, 0.7,f'lead time: {vlt}h', **text_kwargs)
    # else:
    #     ax.text(2.2, -1.9,f'lead time: {vlt}h', **text_kwargs)

    # calculate variable and plot figure
    if pl is not None:
        uwind = ds1.isel(time=rolloutstep, pressure=pl)["x_wind_pl"].to_numpy()
        vwind = ds1.isel(time=rolloutstep, pressure=pl)["y_wind_pl"].to_numpy()
        uwind_era = ds1_era.isel(time=rolloutstep, pressure=pl)["x_wind_pl"].to_numpy()
        vwind_era = ds1_era.isel(time=rolloutstep, pressure=pl)["y_wind_pl"].to_numpy()
        windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
        windspeed_era = np.sqrt(np.square(uwind_era) + np.square(vwind_era))
        t = np.append(windspeed, windspeed_era)
    else:
        uwind = ds1.isel(time=rolloutstep)["x_wind_10m"].to_numpy()
        vwind = ds1.isel(time=rolloutstep)["y_wind_10m"].to_numpy()
        uwind_era = ds1_era.isel(time=rolloutstep)["x_wind_10m"].to_numpy()
        vwind_era = ds1_era.isel(time=rolloutstep)["y_wind_10m"].to_numpy()
        windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
        windspeed_era = np.sqrt(np.square(uwind_era) + np.square(vwind_era))
        t = np.append(windspeed, windspeed_era)
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe = g)

    ax.set_title("\n".join(wrap(title, 40)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    # fig.colorbar(psc, ax=ax)
    return psc
def WindspeedPlotNoBoundaries(fig, ax, ds1, pl, rolloutstep, title, g):
    lat = ds1["latitude"].to_numpy()
    lon = ds1["longitude"].to_numpy()
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)

    # Add Time Stamp
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='black')
    # ax.text(0.35, 0.68, str(ds1.isel(time=0)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=rolloutstep)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    # if g == False:
    #     ax.text(0.35, 0.7,f'lead time: {vlt}h', **text_kwargs)
    # else:
    #     ax.text(2.2, -1.9,f'lead time: {vlt}h', **text_kwargs)

    # calculate variable and plot figure
    if pl is not None:
        uwind = ds1.isel(time=rolloutstep, pressure=pl)["x_wind_pl"].to_numpy()
        vwind = ds1.isel(time=rolloutstep, pressure=pl)["y_wind_pl"].to_numpy()
        t = np.sqrt(np.square(uwind) + np.square(vwind))
    else:
        uwind = ds1.isel(time=rolloutstep)["x_wind_10m"].to_numpy()
        vwind = ds1.isel(time=rolloutstep)["y_wind_10m"].to_numpy()
        t = np.sqrt(np.square(uwind) + np.square(vwind))
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe = g)

    ax.set_title("\n".join(wrap(title, 40)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    # fig.colorbar(psc, ax=ax)
    return psc
def compute_spectra(field):
    """
    Compute spectral variability of a field by wavenumber
    Args:
        field (lat-long numpy array): field to calculate the spectra of


    Returns:
        spectra of field by wavenumber
    """
    field = np.array(field)

    zero, w = SHGLQ(len(field))
    # compute real and imaginary parts of power spectra of field
    coeffs_field = SHExpandGLQ(field, w, zero)


    # Re**2 + Im**2
    coeff_amp = coeffs_field[0,:,:]**2 + coeffs_field[1,:,:]**2


    # sum over meridional direction
    spectra = np.sum(coeff_amp,axis=0)


    return spectra

def getSpectArray(ds1, date, var, lt, pl):
    if var == "windspeed":
        return getSpectArrayWindspeed(ds1, date, var, lt, pl)
    vstart = datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:]), 0, 0, 0)
    vnow = vstart + datetime.timedelta(hours=int(lt))

    if pl is None:
        fcst = ds1.sel(time=vnow)[var].to_numpy().squeeze()
    else:
        fcst = ds1.sel(pressure=pl).sel(time=vnow)[var].to_numpy()
    # Interpolate to match size
    lonlat = np.append(np.expand_dims(ds1["latitude"].to_numpy(), axis=1),np.expand_dims(ds1["longitude"].to_numpy(), axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)
    x = np.linspace(min(pc_lon), max(pc_lon), 129)
    y = np.linspace(min(pc_lat), max(pc_lat), 64)
    xv, yv = np.meshgrid(x, y)
    fcst_arr = scipy.interpolate.griddata(points = pc(lon, lat), values = fcst, xi = (xv, yv), method = "nearest")
    return fcst_arr

def getSpectArrayWindspeed(ds1, date, var, lt, pl):
    vstart = datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:]), 0, 0, 0)
    vnow = vstart + datetime.timedelta(hours=int(lt))

    if pl is None:
        uwind = ds1.sel(time=vnow)["x_wind_10m"].to_numpy()
        vwind = ds1.sel(time=vnow)["y_wind_10m"].to_numpy()
        fcst = np.sqrt(np.square(uwind.squeeze()) + np.square(vwind.squeeze()))
    else:
        uwind = ds1.isel(time=vnow, pressure=pl)["x_wind_pl"].to_numpy()
        vwind = ds1.isel(time=vnow, pressure=pl)["y_wind_pl"].to_numpy()
        fcst = np.sqrt(np.square(uwind) + np.square(vwind))

    # Interpolate to match size
    lonlat = np.append(np.expand_dims(ds1["latitude"].to_numpy(), axis=1),np.expand_dims(ds1["longitude"].to_numpy(), axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)
    x = np.linspace(min(pc_lon), max(pc_lon), 129)
    y = np.linspace(min(pc_lat), max(pc_lat), 64)
    xv, yv = np.meshgrid(x, y)
    fcst_arr = scipy.interpolate.griddata(points = pc(lon, lat), values = fcst, xi = (xv, yv), method = "nearest")
    return fcst_arr

def SpectralPlot(fig, ax, ds1, ds1_truth, var, date, pl, lt, rollout):
    # vstart = datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:]), 0, 0, 0)
    # vnow = vstart + datetime.timedelta(hours=int(lt))
    if rollout != None:
            # plot prediction spectrum
        fcst_arr = getSpectArray(rollout, date, var, lt, pl)
        spectra_nn = compute_spectra(fcst_arr)
        ax.loglog(np.arange(1, spectra_nn.shape[0]), spectra_nn[1:spectra_nn.shape[0]], label = "SG-AIFS resolution 5 rollout", color = "orange")
    # plot prediction spectrum
    fcst_arr = getSpectArray(ds1, date, var, lt, pl)
    spectra_nn = compute_spectra(fcst_arr)
    ax.loglog(np.arange(1, spectra_nn.shape[0]), spectra_nn[1:spectra_nn.shape[0]], label = "SG-AIFS resolution 5", color = "red")

    # plot truth spectrum
    fcst_truth_arr = getSpectArray(ds1_truth, date, var, lt, pl)
    spectra_truth_nn = compute_spectra(fcst_truth_arr)

    psc = ax.loglog(np.arange(1, spectra_truth_nn.shape[0]), spectra_truth_nn[1:spectra_truth_nn.shape[0]], label = f'Ground truth', color='darkred')

    
    # pyl.xlabel('Wavenumber (k)')
    # pyl.ylabel("Spectral variance")
    pyl.legend()

    return psc

def DifferencePlotsNoBoundaries(fig, ax, ds1, ds1_truth, var, pl, rolloutstep, title, g, vmin: Optional[float] = None, vmax: Optional[float] = None):
    if var == "windspeed":
        return WindspeedDifferencePlotNoBoundaries(fig, ax, ds1, ds1_truth, pl, rolloutstep, title, g)
    lat = ds1["latitude"].to_numpy()
    lon = ds1["longitude"].to_numpy()
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    if g == True:
        lon, lat = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)

    # Add Time Stamp
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='black')
    # ax.text(0.35, 0.68, str(ds1.isel(time=0)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=rolloutstep)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    # ax.text(0.35, 0.7,f'lead time: {vlt}h', **text_kwargs)

    # calculate variable and plot figure
    if pl is not None:
        t1 = ds1.isel(time=rolloutstep, pressure=pl)[var].to_numpy()
        t2 = ds1_truth.isel(time=rolloutstep, pressure=pl)[var].to_numpy()
        t = t1 - t2
    else:
        t1 = ds1.isel(time=rolloutstep)[var].to_numpy()
        t2 = ds1_truth.isel(time=rolloutstep)[var].to_numpy()
        t = t1 - t2
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe=g, cmap = "bwr", vmin=vmin, vmax=vmax)

    ax.set_title("\n".join(wrap(title, 40)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    return psc

def WindspeedDifferencePlotNoBoundaries(fig, ax, ds1, ds1_truth, pl, rolloutstep, title, g):
    lat = ds1["latitude"].to_numpy()
    lon = ds1["longitude"].to_numpy()
    lonlat = np.append(np.expand_dims(lat, axis=1),np.expand_dims(lon, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    if g == True:
        lon, lat = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)

    # Add Time Stamp
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='black')
    # ax.text(0.35, 0.68, str(ds1.isel(time=0)["time"].to_numpy())[:19], **text_kwargs)
    vlt = str(((ds1.isel(time=rolloutstep)["time"].to_numpy()-ds1.isel(time=0)["time"].to_numpy())/np.timedelta64(1, 's'))/3600)
    # if g == False:
    #     ax.text(0.35, 0.7,f'lead time: {vlt}h', **text_kwargs)
    # else:
    #     ax.text(2.2, -1.9,f'lead time: {vlt}h', **text_kwargs)

    # calculate variable and plot figure
    if pl is not None:
        uwind = ds1.isel(time=rolloutstep, pressure=pl)["x_wind_pl"].to_numpy()
        vwind = ds1.isel(time=rolloutstep, pressure=pl)["y_wind_pl"].to_numpy()
        t1 = np.sqrt(np.square(uwind) + np.square(vwind))
        uwind = ds1_truth.isel(time=rolloutstep, pressure=pl)["x_wind_pl"].to_numpy()
        vwind = ds1_truth.isel(time=rolloutstep, pressure=pl)["y_wind_pl"].to_numpy()
        t2 = np.sqrt(np.square(uwind) + np.square(vwind))
        t = t1-t2
    else:
        uwind = ds1.isel(time=rolloutstep)["x_wind_10m"].to_numpy()
        vwind = ds1.isel(time=rolloutstep)["y_wind_10m"].to_numpy()
        t1 = np.sqrt(np.square(uwind) + np.square(vwind))
        uwind = ds1_truth.isel(time=rolloutstep)["x_wind_10m"].to_numpy()
        vwind = ds1_truth.isel(time=rolloutstep)["y_wind_10m"].to_numpy()
        t2 = np.sqrt(np.square(uwind) + np.square(vwind))
        t = t1-t2
    psc = scatter_plot(fig, ax, pc_lon, pc_lat, t, s=10, globe = g, cmap = "bwr" )

    ax.set_title("\n".join(wrap(title, 40)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    # fig.colorbar(psc, ax=ax)
    return psc

def HAdifferenceplot(fig, ax, ds1, ds1_truth, var, var_truth, rolloutstep, globe):
    if var == "windspeed":
        return HAdifferenceplotwindspeed(fig, ax, ds1, ds1_truth, rolloutstep, globe)
    difference = (ds1[var] - ds1_truth[var_truth].isel(time=rolloutstep).to_numpy().reshape(789, 789))
    psc = scatter_plot(fig, ax, np.radians((ds1["longitude"]+180)%360 -180), np.radians(ds1["latitude"]), difference, s=10, globe=globe, cmap ="bwr")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    return psc

def HAdifferenceplotwindspeed(fig, ax, ds1, ds1_truth, rolloutstep, globe):
    uwind = ds1["u"]
    vwind = ds1["v"]
    uwind_true = ds1_truth["x_wind_10m"].isel(time=rolloutstep).to_numpy().reshape(789, 789)
    vwind_true = ds1_truth["y_wind_10m"].isel(time=rolloutstep).to_numpy().reshape(789, 789)
    windspeed = np.sqrt(np.square(uwind) + np.square(vwind))
    windspeed_true = np.sqrt(np.square(uwind_true) + np.square(vwind_true))
    psc = scatter_plot(fig, ax, np.radians((ds1["longitude"]+180)%360 -180), np.radians(ds1["latitude"]), (windspeed - windspeed_true), s=10, globe=globe, cmap = "bwr")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    # fig.colorbar(psc, ax=ax)
    return psc