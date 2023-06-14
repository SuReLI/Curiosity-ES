from CMaze import CMaze


env=CMaze(xinit=50, yinit=50, xgoal=200,ygoal=200,width=100,height=100, time_horizon=500)


# env.add_block(180,-10,10,20)
# env.add_block(160,20,50,10)
# env.add_block(160,10,10,20)
# env.add_block(40,10,130,10)
# env.add_block(40,10,10,40)
# env.add_block(10,30,40,10)
# env.add_block(20,-10,10,30)
# # checked

# env.add_block(10,60,10,20)
# env.add_block(10,70,30,10)
# env.add_block(30,70,10,60)
# env.add_block(-10,100,30,10)
# env.add_block(10,120,30,10)
# env.add_block(10,120,10,90)
# # checked

# env.add_block(30,50,30,10)
# env.add_block(50,50,10,90)
# env.add_block(30,140,30,10)
# env.add_block(30,140,10,50)
# env.add_block(50,160,10,50)
# env.add_block(50,160,30,10)
# env.add_block(70,30,10,140)
# # checked

# env.add_block(100,10,10,180)
# env.add_block(90,30,30,10)
# env.add_block(90,70,30,10)
# env.add_block(90,110,30,10)
# env.add_block(90,150,30,10)
# env.add_block(130,30,10,180)
# # checked
# env.add_block(70,50,20,10)
# env.add_block(70,90,20,10)
# env.add_block(70,130,20,10)
# env.add_block(120,50,20,10)
# env.add_block(120,90,20,10)
# env.add_block(120,130,20,10)
# # checked
# env.add_block(150,40,10,150)
# env.add_block(180,40,10,150)
# env.add_block(150,120,40,10)



# env.reset()

# env.render('C')

env.save("easy")
