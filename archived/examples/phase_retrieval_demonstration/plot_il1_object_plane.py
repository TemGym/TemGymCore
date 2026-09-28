"""Show the moving detector-conjugate plane in the objective-to-IL1 gap."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from demonstration import (HERE, COLORS, microscope, optics, column_matrix, Plane,
                           Ray, run_to_end, _compute_cumulative_rotation, save)


def main():
    model = microscope()
    saved = json.loads((HERE/'calibration.json').read_text())
    column, reference = optics(model, 0.)
    objective, il1 = column[:2]
    zs = float(model.components['sample'].z)
    zo, zi = float(objective.z), float(il1.z)
    z_image = reference['image_planes'][0]['image_z_m']
    # Remove the fixed specimen-to-IL1-entrance transfer from each calibrated
    # full matrix. This leaves the downstream transfer including IL1 itself.
    prefix = np.asarray(column_matrix((objective, Plane(z=zi)),
                        source_z=zs, voltage=model.voltage))
    records = [reference, *saved['settings']]
    colors = ['#505a66', *COLORS]
    results = []
    rays = Ray(x=np.zeros(3), y=np.zeros(3), dx=np.array([-1e-3, 0., 1e-3]),
               dy=np.zeros(3), z=np.full(3,zs), voltage=np.full(3,model.voltage),
               pathlength=np.zeros(3))
    after = run_to_end(rays, (objective,))
    angle = float(_compute_cumulative_rotation((objective,), voltage=model.voltage,
                  include_input_rays=False)[-1])
    slope = np.asarray(after.dx)*np.cos(angle)+np.asarray(after.dy)*np.sin(angle)
    xx0 = np.asarray(after.x)*np.cos(angle)+np.asarray(after.y)*np.sin(angle)
    np.testing.assert_allclose(xx0+(z_image-zo)*slope, 0., atol=1e-15)
    fig, axes = plt.subplots(1,5,figsize=(16,7.8),sharey=True)
    fig.subplots_adjust(left=.065,right=.985,top=.80,bottom=.20,wspace=.17)
    zz = np.linspace(zo, zi, 300)
    for ax, rec, color in zip(axes, records, colors):
        downstream = np.asarray(rec['matrix']) @ np.linalg.inv(prefix)
        A, B = downstream[:2,:2], downstream[:2,2:4]
        q = np.linalg.solve(A, B)
        offset = float(np.trace(q)/2)
        np.testing.assert_allclose(q, offset*np.eye(2), atol=1e-12)
        z_conjugate = zi+offset
        assert zo < z_conjugate < zi
        # Independently verify the actual native downstream column images this
        # plane at the fixed detector: its position-angle block must vanish.
        changed = model.with_currents(IL1=rec['current_A'])
        downstream_column = tuple(c for c in changed.column if float(c.z)>=zi)
        check = np.asarray(column_matrix(downstream_column, source_z=z_conjugate,
                                         voltage=model.voltage))
        np.testing.assert_allclose(check[:2,2:4], 0., atol=1e-10)
        delta = (z_conjugate-z_image)*1e3
        # Solid: physical rays from one specimen point through the objective.
        actual = xx0[None,:]+(zz[:,None]-zo)*slope[None,:]
        ax.plot(actual*1e6, (zz-zs)*1e3, color='0.63', lw=1.5, alpha=.8)
        # Dashed: detector-conjugate rays, back-projected through the downstream
        # optics, drawn only in the free-space gap. Same slope range in all panels.
        selected = (zz[:,None]-z_conjugate)*slope[None,:]
        ax.plot(selected*1e6, (zz-zs)*1e3, color=color, ls='--', lw=1.6)
        ax.axhline((zo-zs)*1e3, color='0.25', lw=2)
        ax.axhline((zi-zs)*1e3, color='0.25', lw=2)
        ax.axhline((z_image-zs)*1e3, color='#7c3aed', ls=':', lw=1.8)
        ax.axhline((z_conjugate-zs)*1e3, color=color, ls='--', lw=1.3)
        ax.scatter(0,(z_conjugate-zs)*1e3,s=50,color=color,zorder=8)
        if abs(delta)>1e-8:
            ax.annotate('',xy=(2.65,(z_conjugate-zs)*1e3),
                        xytext=(2.65,(z_image-zs)*1e3),
                        arrowprops=dict(arrowstyle='<->',color=color,lw=1.5))
        ax.text(.04,.975,'Objective lens',transform=ax.transAxes,fontsize=9,va='top')
        ax.text(.04,.035,'IL1',transform=ax.transAxes,fontsize=10,va='bottom')
        ax.set_title(('Focused reference' if rec['delta']==0 else f"IL1 {rec['delta']:+.0%}")+
                     f"\n{rec['current_A']:.3f} A",fontsize=13,color=color,pad=12)
        ax.text(.5,-.14,f"Plane shift: {delta:+.2f} mm",transform=ax.transAxes,
                ha='center',fontsize=11,color=color,weight='bold')
        ax.set(xlim=(-3.1,3.1),ylim=((zi-zs)*1e3+9,-7),xlabel='x (µm)',xticks=[-2,0,2])
        ax.spines[['top','right']].set_visible(False)
        results.append(dict(IL1_delta=rec['delta'],IL1_current_A=rec['current_A'],
                       detector_conjugate_z_m=z_conjugate,
                       objective_image_z_m=z_image,plane_shift_mm=delta,
                       native_max_abs_B_m=float(np.abs(check[:2,2:4]).max())))
    axes[0].set_ylabel('z from specimen (mm)')
    fig.suptitle('IL1 moves the plane brought into focus on the fixed detector',
                 fontsize=20,weight='bold',y=.96)
    fig.text(.065,.875,'Objective → IL1 gap: defocus is the separation from the fixed objective image plane',
             fontsize=13,color='#475569')
    handles=[Line2D([],[],color='0.63',lw=1.5,label='Objective rays (unchanged)'),
             Line2D([],[],color='#31688e',ls='--',lw=1.5,label='Back-projected detector rays'),
             Line2D([],[],color='#7c3aed',ls=':',lw=1.8,label='Fixed objective image plane')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.02),ncol=3,frameon=False,fontsize=11)
    save(fig,'03d_IL1_object_plane_defocus')
    (HERE/'IL1_object_plane_defocus.json').write_text(json.dumps(dict(
        sign_convention='Positive shift is downstream towards IL1.',
        explanation='The objective image remains fixed. The detector-conjugate plane moves with IL1.',
        settings=results),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
