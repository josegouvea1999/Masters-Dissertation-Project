import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { NavigationContainer } from '@react-navigation/native';
import LoginScreen from './screens/LoginScreen';
import HomeScreen from './screens/HomeScreen';
import AddScreen from './screens/AddScreen';

const Stack = createNativeStackNavigator();
function Navigation() {
    return (
        <NavigationContainer>
            <Stack.Navigator>
                <Stack.Screen name="Login" component={LoginScreen} options={{headerShown:false}}/>
                <Stack.Screen name="Main" component={HomeScreen} options={{headerShown:false}}/>
                <Stack.Screen name="Add" component={AddScreen} options={{headerShown:false}}/>
            </Stack.Navigator>
        </NavigationContainer>
    )
}

export default Navigation